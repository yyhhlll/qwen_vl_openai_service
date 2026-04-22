from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .protocol import ChatCompletionRequest
except ImportError:
    from protocol import ChatCompletionRequest


API_KEY = os.getenv("API_KEY", "1234")
BACKEND_URLS = [
    item.strip()
    for item in os.getenv(
        "BACKEND_URLS",
        "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004",
    ).split(",")
    if item.strip()
]
TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=3600.0)

if not BACKEND_URLS:
    raise RuntimeError("BACKEND_URLS must not be empty.")

_backend_lock = asyncio.Lock()
_client = httpx.AsyncClient(timeout=TIMEOUT)
_backend_inflight: dict[str, int] = {backend: 0 for backend in BACKEND_URLS}

app = FastAPI()


def _verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


async def _pick_backend() -> str:
    async with _backend_lock:
        return min(BACKEND_URLS, key=lambda backend: (_backend_inflight[backend], backend))


async def _mark_backend_start(backend: str) -> None:
    async with _backend_lock:
        _backend_inflight[backend] += 1


async def _mark_backend_done(backend: str) -> None:
    async with _backend_lock:
        _backend_inflight[backend] = max(_backend_inflight[backend] - 1, 0)


@app.get("/health")
async def health() -> dict[str, object]:
    statuses: list[dict[str, object]] = []
    for backend in BACKEND_URLS:
        try:
            response = await _client.get(f"{backend}/health")
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            statuses.append(
                {
                    "backend": backend,
                    "status_code": response.status_code,
                    "ok": response.status_code == 200,
                    "proxy_inflight": _backend_inflight[backend],
                    "backend_health": payload,
                }
            )
        except Exception as exc:
            statuses.append(
                {
                    "backend": backend,
                    "status_code": None,
                    "ok": False,
                    "error": str(exc),
                    "proxy_inflight": _backend_inflight[backend],
                }
            )
    return {"ok": any(item["ok"] for item in statuses), "backends": statuses}


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> Response:
    _verify_api_key(authorization)
    backend = await _pick_backend()
    await _mark_backend_start(backend)
    try:
        response = await _client.get(
            f"{backend}/v1/models",
            headers={"Authorization": authorization or ""},
        )
    finally:
        await _mark_backend_done(backend)
    return JSONResponse(status_code=response.status_code, content=response.json())


@app.post("/v1/chat/completions")
async def chat_completions(
    payload: ChatCompletionRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
):
    _verify_api_key(authorization)
    backend = await _pick_backend()
    await _mark_backend_start(backend)
    headers = {
        "Authorization": authorization or "",
        "Content-Type": "application/json",
    }
    body = payload.model_dump()

    if payload.stream:
        request = _client.build_request(
            "POST",
            f"{backend}/v1/chat/completions",
            headers=headers,
            json=body,
        )
        upstream = await _client.send(request, stream=True)

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    if await raw_request.is_disconnected():
                        break
                    if chunk:
                        yield chunk
            finally:
                await upstream.aclose()
                await _mark_backend_done(backend)

        return StreamingResponse(
            _iter(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        response = await _client.post(
            f"{backend}/v1/chat/completions",
            headers=headers,
            json=body,
        )
    finally:
        await _mark_backend_done(backend)
    return JSONResponse(status_code=response.status_code, content=response.json())
