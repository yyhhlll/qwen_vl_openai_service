from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

try:
    from .protocol import ChatCompletionRequest
except ImportError:
    from protocol import ChatCompletionRequest


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
LOGGER = logging.getLogger("qwen35_proxy")

API_KEY = os.getenv("API_KEY", "1234")
TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "Qwen3.5-0.6B")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "Qwen3.5-4B")
DEFAULT_VISION_BACKENDS = (
    "http://127.0.0.1:8001,http://127.0.0.1:8002,"
    "http://127.0.0.1:8003,http://127.0.0.1:8004"
)
DEFAULT_TEXT_BACKENDS = "http://127.0.0.1:12000"
TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=3600.0)


def _split_urls(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_urls(*names: str, default: str) -> list[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            urls = _split_urls(value)
            if not urls:
                raise RuntimeError(f"{name} must not be empty when set.")
            return urls
    return _split_urls(default)


VISION_BACKEND_URLS = _env_urls(
    "VISION_BACKEND_URLS",
    "BACKEND_URLS_4B",
    "BACKEND_URLS",
    default=DEFAULT_VISION_BACKENDS,
)
TEXT_BACKEND_URLS = _env_urls("TEXT_BACKEND_URLS", default=DEFAULT_TEXT_BACKENDS)

_client = httpx.AsyncClient(timeout=TIMEOUT)


class BackendPool:
    def __init__(self, name: str, urls: list[str], model_name: str) -> None:
        if not urls:
            raise RuntimeError(f"{name} backend URLs must not be empty.")
        self.name = name
        self.urls = urls
        self.model_name = model_name
        self._lock = asyncio.Lock()
        self._inflight: dict[str, int] = {backend: 0 for backend in urls}

    async def pick(self) -> str:
        async with self._lock:
            return min(self.urls, key=lambda backend: (self._inflight[backend], backend))

    async def mark_start(self, backend: str) -> None:
        async with self._lock:
            self._inflight[backend] += 1

    async def mark_done(self, backend: str) -> None:
        async with self._lock:
            self._inflight[backend] = max(self._inflight[backend] - 1, 0)

    def inflight(self, backend: str) -> int:
        return self._inflight[backend]


TEXT_POOL = BackendPool("text_0_6b", TEXT_BACKEND_URLS, TEXT_MODEL_NAME)
VISION_POOL = BackendPool("vision_4b", VISION_BACKEND_URLS, VISION_MODEL_NAME)

app = FastAPI()


def _verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def _headers(authorization: str | None) -> dict[str, str]:
    return {
        "Authorization": authorization or "",
        "Content-Type": "application/json",
    }


TEXT_BACKEND_ALLOWED_KEYS = {"model", "messages", "max_tokens"}


def _text_content_to_string(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    text_parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "\n".join(text_parts)


def _messages_for_text_backend(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    normalized: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        text_message = {
            "role": message.get("role"),
            "content": _text_content_to_string(message.get("content")),
        }
        normalized.append(text_message)
    return normalized


def _body_for_backend(body: dict[str, Any], model_name: str, *, text_backend: bool = False) -> dict[str, Any]:
    if text_backend:
        backend_body = {key: deepcopy(body[key]) for key in TEXT_BACKEND_ALLOWED_KEYS if key in body}
        if "messages" in backend_body:
            backend_body["messages"] = _messages_for_text_backend(backend_body["messages"])
    else:
        backend_body = deepcopy(body)
    backend_body["model"] = model_name
    return backend_body


def _normalize_response_model(payload: Any, requested_model: str | None) -> Any:
    if requested_model and isinstance(payload, dict) and payload.get("object") == "chat.completion":
        payload = deepcopy(payload)
        payload["model"] = requested_model
    return payload


def _contains_image_content(content: Any) -> bool:
    if isinstance(content, list):
        return any(isinstance(part, dict) and part.get("type") == "image_url" for part in content)
    return False


def request_has_image(body: dict[str, Any]) -> bool:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        return False
    return any(isinstance(message, dict) and _contains_image_content(message.get("content")) for message in messages)


def _extract_assistant_content(response_payload: Any) -> str | None:
    if not isinstance(response_payload, dict):
        return None
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _string_verdict(value: str) -> bool | None:
    normalized = value.strip().lower()
    safe_values = {"合规", "安全", "safe", "compliant", "true", "yes", "pass", "passed"}
    unsafe_values = {
        "不合规",
        "不安全",
        "unsafe",
        "noncompliant",
        "non-compliant",
        "not compliant",
        "false",
        "no",
        "fail",
        "failed",
    }
    if normalized in safe_values:
        return True
    if normalized in unsafe_values:
        return False
    return None


def _json_verdict(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _string_verdict(value)
    return None


def _json_candidates(content: str) -> list[str]:
    candidates = [content]
    import re

    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE))
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(content[start : end + 1])
    return candidates


def _categories_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return bool(normalized and normalized not in {"none", "null", "无", "[]"})
    return bool(value)


def _json_is_confidently_compliant(parsed: dict[str, Any]) -> bool | None:
    for key in ("compliant", "is_compliant", "is_safty", "is_safety", "safe"):
        if key in parsed:
            verdict = _json_verdict(parsed[key])
            if verdict is not None:
                return verdict

    for key in ("result", "status", "safety"):
        if key in parsed:
            verdict = _json_verdict(parsed[key])
            if verdict is not None:
                return verdict

    for key in ("category", "categories", "unsafy_class", "unsafe_class", "risk_class"):
        if key in parsed and _categories_present(parsed[key]):
            return False
    return None


def _guard_text_verdict(content: str) -> bool | None:
    import re

    safety_match = re.search(r"^\s*Safety\s*:\s*(Safe|Unsafe)\s*$", content, re.IGNORECASE | re.MULTILINE)
    if safety_match:
        return safety_match.group(1).lower() == "safe"
    verdict_match = re.search(r"^\s*(?:Result|Status|Verdict|结论|结果)\s*[:：]\s*(合规|不合规|安全|不安全|safe|unsafe|compliant|non[- ]?compliant)\s*$", content, re.IGNORECASE | re.MULTILINE)
    if verdict_match:
        return _string_verdict(verdict_match.group(1))
    return _string_verdict(content)


def is_confidently_compliant(response_payload: Any) -> bool:
    content = _extract_assistant_content(response_payload)
    if content is None:
        return False
    stripped = content.strip()
    if not stripped:
        return False

    for candidate in _json_candidates(stripped):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            verdict = _json_is_confidently_compliant(parsed)
            if verdict is not None:
                return verdict is True

    verdict = _guard_text_verdict(stripped)
    return verdict is True


async def _pool_health(pool: BackendPool) -> dict[str, object]:
    statuses: list[dict[str, object]] = []
    for backend in pool.urls:
        try:
            response = await _client.get(f"{backend}/health")
            payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            statuses.append(
                {
                    "backend": backend,
                    "status_code": response.status_code,
                    "ok": response.status_code == 200,
                    "proxy_inflight": pool.inflight(backend),
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
                    "proxy_inflight": pool.inflight(backend),
                }
            )
    return {
        "name": pool.name,
        "model": pool.model_name,
        "ok": any(item["ok"] for item in statuses),
        "backends": statuses,
    }


@app.get("/health")
async def health() -> dict[str, object]:
    text_health, vision_health = await asyncio.gather(
        _pool_health(TEXT_POOL),
        _pool_health(VISION_POOL),
    )
    return {
        "ok": bool(text_health["ok"] or vision_health["ok"]),
        "pools": {
            "text": text_health,
            "vision": vision_health,
        },
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> Response:
    _verify_api_key(authorization)
    models = []
    for model_name in (TEXT_MODEL_NAME, VISION_MODEL_NAME):
        if model_name not in [item["id"] for item in models]:
            models.append({"id": model_name, "object": "model", "owned_by": "local"})
    return JSONResponse(content={"object": "list", "data": models})


async def _post_to_pool(
    pool: BackendPool,
    path: str,
    authorization: str | None,
    body: dict[str, Any],
) -> tuple[httpx.Response, str, float]:
    backend = await pool.pick()
    await pool.mark_start(backend)
    started = time.perf_counter()
    LOGGER.info(
        "proxy_upstream_start pool=%s backend=%s path=%s internal_model=%s",
        pool.name,
        backend,
        path,
        pool.model_name,
    )
    try:
        response = await _client.post(
            f"{backend}{path}",
            headers=_headers(authorization),
            json=_body_for_backend(body, pool.model_name, text_backend=pool is TEXT_POOL),
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        LOGGER.info(
            "proxy_upstream_done pool=%s backend=%s status=%s elapsed_ms=%.1f internal_model=%s",
            pool.name,
            backend,
            response.status_code,
            elapsed_ms,
            pool.model_name,
        )
        return response, backend, elapsed_ms
    finally:
        await pool.mark_done(backend)


async def _stream_from_pool(
    pool: BackendPool,
    path: str,
    authorization: str | None,
    body: dict[str, Any],
) -> tuple[httpx.Response, str]:
    backend = await pool.pick()
    await pool.mark_start(backend)
    LOGGER.info(
        "proxy_upstream pool=%s backend=%s path=%s internal_model=%s stream=true",
        pool.name,
        backend,
        path,
        pool.model_name,
    )
    request = _client.build_request(
        "POST",
        f"{backend}{path}",
        headers=_headers(authorization),
        json=_body_for_backend(body, pool.model_name, text_backend=pool is TEXT_POOL),
    )
    try:
        upstream = await _client.send(request, stream=True)
    except Exception:
        await pool.mark_done(backend)
        raise
    return upstream, backend


def _json_or_none(response: httpx.Response) -> Any | None:
    try:
        return response.json()
    except Exception:
        return None


async def _call_vision_non_streaming(
    body: dict[str, Any],
    requested_model: str | None,
    authorization: str | None,
    *,
    route: str,
    extra_headers: dict[str, str] | None = None,
) -> JSONResponse:
    response, backend, elapsed_ms = await _post_to_pool(VISION_POOL, "/v1/chat/completions", authorization, body)
    headers = {
        "X-Qwen-Route": route,
        "X-Qwen-Vision-Backend": backend,
        "X-Qwen-Vision-Elapsed-Ms": f"{elapsed_ms:.1f}",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload = _json_or_none(response)
    if payload is None:
        return JSONResponse(status_code=response.status_code, content={"detail": response.text}, headers=headers)
    return JSONResponse(
        status_code=response.status_code,
        content=_normalize_response_model(payload, requested_model),
        headers=headers,
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    raw_request: Request,
    authorization: str | None = Header(default=None),
):
    _verify_api_key(authorization)
    try:
        body = await raw_request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")

    try:
        parsed_request = ChatCompletionRequest.model_validate(body)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    requested_model = body.get("model") if isinstance(body.get("model"), str) else parsed_request.model

    if parsed_request.stream:
        LOGGER.info("route_decision=stream_direct_4b requested_model=%s", requested_model)
        upstream, backend = await _stream_from_pool(
            VISION_POOL,
            "/v1/chat/completions",
            authorization,
            body,
        )

        async def _iter() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_bytes():
                    if await raw_request.is_disconnected():
                        break
                    if chunk:
                        yield chunk
            finally:
                await upstream.aclose()
                await VISION_POOL.mark_done(backend)

        return StreamingResponse(
            _iter(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    if request_has_image(body):
        LOGGER.info("route_decision=image_direct_4b requested_model=%s", requested_model)
        return await _call_vision_non_streaming(
            body,
            requested_model,
            authorization,
            route="image_direct_4b",
        )

    text_headers: dict[str, str] = {}
    try:
        LOGGER.info("route_decision=text_first_0_6b requested_model=%s", requested_model)
        text_response, text_backend, text_elapsed_ms = await _post_to_pool(TEXT_POOL, "/v1/chat/completions", authorization, body)
        text_headers = {
            "X-Qwen-Text-Backend": text_backend,
            "X-Qwen-Text-Elapsed-Ms": f"{text_elapsed_ms:.1f}",
        }
        if text_response.status_code == 200:
            text_payload = _json_or_none(text_response)
            if is_confidently_compliant(text_payload):
                LOGGER.info("route_result=text_0_6b_only requested_model=%s elapsed_ms=%.1f", requested_model, text_elapsed_ms)
                headers = {"X-Qwen-Route": "text_0_6b_only", **text_headers}
                return JSONResponse(
                    status_code=text_response.status_code,
                    content=_normalize_response_model(text_payload, requested_model),
                    headers=headers,
                )
            LOGGER.info("route_escalate=text_0_6b_not_confident requested_model=%s elapsed_ms=%.1f", requested_model, text_elapsed_ms)
        else:
            LOGGER.info(
                "route_escalate=text_0_6b_status status_code=%s requested_model=%s elapsed_ms=%.1f",
                text_response.status_code,
                requested_model,
                text_elapsed_ms,
            )
    except Exception as exc:
        LOGGER.info("route_escalate=text_0_6b_exception error=%s requested_model=%s", exc, requested_model)

    LOGGER.info("route_result=text_0_6b_then_4b requested_model=%s", requested_model)
    return await _call_vision_non_streaming(
        body,
        requested_model,
        authorization,
        route="text_0_6b_then_4b",
        extra_headers=text_headers,
    )
