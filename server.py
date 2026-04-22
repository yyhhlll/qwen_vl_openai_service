from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .engine import TransformersVLEngine
    from .protocol import (
        ChatCompletionChunk,
        ChatCompletionChunkChoice,
        ChatCompletionChoice,
        ChatCompletionChoiceDelta,
        ChatCompletionRequest,
        ChatCompletionResponse,
        Usage,
    )
    from .scheduler import Scheduler
except ImportError:
    from engine import TransformersVLEngine
    from protocol import (
        ChatCompletionChunk,
        ChatCompletionChunkChoice,
        ChatCompletionChoice,
        ChatCompletionChoiceDelta,
        ChatCompletionRequest,
        ChatCompletionResponse,
        Usage,
    )
    from scheduler import Scheduler

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3.5-27B-VL")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3.5-27B")
API_KEY = os.getenv("API_KEY", "1234")


def _env_flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_max_batch_size() -> int:
    raw_max_batch_size = os.getenv("MAX_BATCH_SIZE")
    raw_max_running = os.getenv("MAX_RUNNING")

    if raw_max_batch_size is not None:
        value = raw_max_batch_size
    elif raw_max_running is not None:
        value = raw_max_running
    else:
        value = "4"

    max_batch_size = int(value)
    if max_batch_size < 1:
        raise ValueError("MAX_BATCH_SIZE/MAX_RUNNING must be >= 1")
    return max_batch_size


MAX_BATCH_SIZE = _resolve_max_batch_size()
BATCH_WAIT_MS = float(os.getenv("BATCH_WAIT_MS", "50"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "0")) or None
GPU_MEMORY_CLEANUP_INTERVAL = max(
    int(os.getenv("GPU_MEMORY_CLEANUP_INTERVAL", "32")),
    0,
)
OFFLINE_MODE = _env_flag("OFFLINE_MODE", "1")
ALLOW_REMOTE_IMAGE_URLS = _env_flag("ALLOW_REMOTE_IMAGE_URLS", "0")

engine = TransformersVLEngine(
    MODEL_PATH,
    max_model_len=MAX_MODEL_LEN,
    memory_cleanup_interval=GPU_MEMORY_CLEANUP_INTERVAL,
    offline_mode=OFFLINE_MODE,
    allow_remote_image_urls=ALLOW_REMOTE_IMAGE_URLS,
)
scheduler = Scheduler(engine, max_batch_size=MAX_BATCH_SIZE, batch_wait_ms=BATCH_WAIT_MS)


def _verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token.")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await scheduler.start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, object]:
    model_device = None
    if engine.model is not None:
        try:
            model_device = str(next(engine.model.parameters()).device)
        except Exception:
            model_device = None

    return {
        "ok": True,
        "model": MODEL_NAME,
        "max_model_len": engine.max_model_len if engine.model is not None else None,
        "binding": {
            "device_map": engine.device_map,
            "hip_visible_devices": os.getenv("HIP_VISIBLE_DEVICES"),
            "rocr_visible_devices": os.getenv("ROCR_VISIBLE_DEVICES"),
            "model_device": model_device,
        },
        "offline": {
            "offline_mode": engine.offline_mode,
            "allow_remote_image_urls": engine.allow_remote_image_urls,
        },
        "scheduler": {
            "queue_size": scheduler._queue.qsize(),
            "max_batch_size": scheduler.max_batch_size,
            "batch_wait_ms": scheduler.batch_wait_ms,
        },
        "memory_cleanup_interval": engine.memory_cleanup_interval,
    }


@app.get("/v1/models")
async def list_models(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _verify_api_key(authorization)
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
):
    _verify_api_key(authorization)
    # 放开 model 字段校验：后端只服务本地加载的那一个模型，
    # 客户端传什么 model 名都接受，响应里原样回传，便于上层网关/评测工具
    # 用任意别名调用本服务。
    request_model = request.model or MODEL_NAME

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    state = await engine.build_state(request_id, request)
    await scheduler.submit(state)
    await state.result_future

    if state.error:
        raise HTTPException(status_code=500, detail=state.error)

    now = int(time.time())
    usage = Usage(
        prompt_tokens=state.prompt_token_count,
        completion_tokens=state.completion_token_count,
        total_tokens=state.prompt_token_count + state.completion_token_count,
    )

    if request.stream:
        return StreamingResponse(
            _fake_stream(request_id, now, state.generated_text, usage, request_model),
            media_type="text/event-stream",
        )

    return JSONResponse(
        ChatCompletionResponse(
            id=request_id,
            created=now,
            model=request_model,
            choices=[
                ChatCompletionChoice(
                    message={"role": "assistant", "content": state.generated_text},
                    finish_reason="stop",
                )
            ],
            usage=usage,
        ).model_dump()
    )


async def _fake_stream(
    request_id: str,
    created: int,
    text: str,
    usage: Usage,
    model: str = MODEL_NAME,
) -> AsyncIterator[str]:
    """Yield the fully-generated text as SSE chunks, simulating streaming."""
    # Role chunk
    first = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChoiceDelta(role="assistant", content=""),
            )
        ],
    )
    yield f"data: {json.dumps(first.model_dump(), ensure_ascii=False)}\n\n"

    # Content chunks — split by character groups for a natural feel
    chunk_size = max(1, len(text) // 20) if len(text) > 20 else 1
    pos = 0
    while pos < len(text):
        segment = text[pos : pos + chunk_size]
        pos += chunk_size
        chunk = ChatCompletionChunk(
            id=request_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(
                    delta=ChatCompletionChoiceDelta(content=segment),
                )
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump(), ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.01)

    # Final chunk with finish_reason and usage
    final = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                delta=ChatCompletionChoiceDelta(content=""),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {json.dumps(final.model_dump(), ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
