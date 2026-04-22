from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

try:
    from .protocol import ChatCompletionRequest
except ImportError:
    from protocol import ChatCompletionRequest


@dataclass
class RequestState:
    request_id: str
    request: ChatCompletionRequest
    prompt_text: str
    images: list[object]
    generated_text: str = ""
    prompt_token_count: int = 0
    completion_token_count: int = 0
    finished: bool = False
    error: str | None = None
    result_future: asyncio.Future[None] = field(default_factory=lambda: asyncio.get_running_loop().create_future())
