from __future__ import annotations

import asyncio
import base64
import gc
import io
import os
from pathlib import Path
from typing import Any
from urllib.request import urlopen

try:
    from .protocol import ChatMessage
    from .state import RequestState
except ImportError:
    from protocol import ChatMessage
    from state import RequestState


class TransformersVLEngine:
    """Single-process Transformers VL engine with batch inference support."""

    DEFAULT_MAX_TOKENS = 2048

    def __init__(
        self,
        model_path: str,
        device_map: str = "auto",
        max_model_len: int | None = None,
        memory_cleanup_interval: int = 32,
        offline_mode: bool = True,
        allow_remote_image_urls: bool = False,
        remote_image_timeout_seconds: float = 60.0,
    ) -> None:
        self.model_path = model_path
        self.device_map = device_map
        self._user_max_model_len = max_model_len
        self.memory_cleanup_interval = max(memory_cleanup_interval, 0)
        self.offline_mode = offline_mode
        self.allow_remote_image_urls = allow_remote_image_urls
        self.remote_image_timeout_seconds = max(remote_image_timeout_seconds, 0.1)
        self.max_model_len: int = 0
        self.processor = None
        self.model = None
        self._load_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._batches_since_cleanup = 0

    async def ensure_loaded(self) -> None:
        async with self._load_lock:
            if self.model is not None:
                return
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            if self.offline_mode:
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

            pretrained_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "local_files_only": self.offline_mode,
            }

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                **pretrained_kwargs,
            )
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_path,
                **pretrained_kwargs,
                dtype=torch.float16,
                attn_implementation="sdpa",
                device_map=self.device_map,
            ).eval()
            if getattr(self.model, "generation_config", None) is not None:
                self.model.generation_config.thinking = False

            config_max = getattr(
                self.model.config, "max_position_embeddings", None
            ) or getattr(self.model.config, "max_length", 32768)
            self.max_model_len = self._user_max_model_len or config_max

    def _load_image(self, source: str) -> Any:
        from PIL import Image

        if source.startswith("data:image/"):
            _, payload = source.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(payload))).convert("RGB")
        if source.startswith(("http://", "https://")):
            if not self.allow_remote_image_urls:
                raise ValueError(
                    "Remote image URLs are disabled by configuration. "
                    "Use a local file path or a data:image base64 URL."
                )
            with urlopen(source, timeout=self.remote_image_timeout_seconds) as response:
                return Image.open(io.BytesIO(response.read())).convert("RGB")
        return Image.open(Path(source)).convert("RGB")

    def _normalize_messages(
        self,
        messages: list[ChatMessage],
    ) -> tuple[list[dict[str, Any]], list]:
        normalized: list[dict[str, Any]] = []
        images: list = []
        for message in messages:
            if isinstance(message.content, str):
                normalized.append({"role": message.role, "content": message.content})
                continue
            parts: list[dict[str, Any]] = []
            for item in message.content:
                if item.type == "text" and item.text is not None:
                    parts.append({"type": "text", "text": item.text})
                elif item.type == "image_url" and item.image_url is not None:
                    image = self._load_image(item.image_url.url)
                    images.append(image)
                    parts.append({"type": "image"})
            normalized.append({"role": message.role, "content": parts})
        return normalized, images

    async def build_state(self, request_id: str, request: Any) -> RequestState:
        await self.ensure_loaded()
        assert self.processor is not None
        # Keep blocking image fetch/decode off the event loop so one slow image
        # source does not stall the whole backend process.
        normalized_messages, images = await asyncio.to_thread(
            self._normalize_messages,
            request.messages,
        )
        try:
            prompt_text = self.processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Keep compatibility with processor versions that do not expose
            # the enable_thinking template kwarg.
            prompt_text = self.processor.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return RequestState(
            request_id=request_id,
            request=request,
            prompt_text=prompt_text,
            images=images,
        )

    def _release_batch_state(self, batch: list[RequestState]) -> None:
        for state in batch:
            state.images.clear()
            state.prompt_text = ""

    async def _maybe_cleanup_memory(self, force: bool = False) -> None:
        if not force:
            if self.memory_cleanup_interval <= 0:
                return
            self._batches_since_cleanup += 1
            if self._batches_since_cleanup < self.memory_cleanup_interval:
                return

        async with self._cleanup_lock:
            if not force:
                if self._batches_since_cleanup < self.memory_cleanup_interval:
                    return
            self._batches_since_cleanup = 0

            try:
                gc.collect()

                import torch

                if not hasattr(torch, "cuda") or not torch.cuda.is_available():
                    return

                torch.cuda.empty_cache()
            except Exception:
                return

    async def generate_batch(self, batch: list[RequestState]) -> None:
        """Run batch inference — generate all requests in one model.generate() call."""
        await self.ensure_loaded()
        assert self.processor is not None
        assert self.model is not None

        import torch

        inputs: dict[str, Any] | None = None
        input_ids = None
        generation_kwargs: dict[str, Any] | None = None
        output_ids = None
        force_cleanup = False

        try:
            texts = [s.prompt_text for s in batch]

            # 关键修复：只有当批内存在图像时才把 images 传给 processor。
            # 若整批纯文本，传 images=None / [None,...] 会让 Qwen VL processor
            # 走进视觉分支并抛 "only a single or a list of entries is supported
            # but got type=<class 'NoneType'>"。
            has_any_image = any(s.images for s in batch)
            processor_kwargs: dict[str, Any] = {
                "text": texts,
                "return_tensors": "pt",
                "padding": True,
            }
            if has_any_image:
                # 注意：若批内"部分有图、部分无图"，Qwen processor 对
                # list 内混 None 同样不接受——这种混合批需要在 scheduler
                # 侧按 has_images 分桶。当前修复仅解决"全批纯文本"。
                processor_kwargs["images"] = [
                    s.images if s.images else None for s in batch
                ]

            inputs = self.processor(**processor_kwargs)

            input_ids = inputs["input_ids"]
            for i, state in enumerate(batch):
                pad_token_id = self.processor.tokenizer.pad_token_id or 0
                state.prompt_token_count = int(
                    (input_ids[i] != pad_token_id).sum().item()
                )

            model_device = next(self.model.parameters()).device
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }

            # Use generation params from first request for the batch
            first = batch[0].request
            do_sample = first.temperature > 0
            generation_kwargs = {
                **inputs,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = max(first.temperature, 1e-5)
                if first.top_p < 1.0:
                    generation_kwargs["top_p"] = first.top_p

            # Dynamic max_new_tokens: cap to remaining context window
            max_input_len = max(s.prompt_token_count for s in batch)
            remaining = self.max_model_len - max_input_len
            if remaining <= 0:
                for state in batch:
                    state.error = (
                        f"Input too long ({max_input_len} tokens) for model context "
                        f"({self.max_model_len} tokens), no room for generation."
                    )
                    state.finished = True
                force_cleanup = True
                return

            max_tokens_values = [
                s.request.max_tokens
                if s.request.max_tokens is not None
                else self.DEFAULT_MAX_TOKENS
                for s in batch
            ]
            requested = max(max_tokens_values)
            if requested > remaining:
                if any(s.request.max_tokens is not None for s in batch):
                    for state in batch:
                        state.error = (
                            f"Input ({max_input_len} tokens) + requested max_tokens ({requested}) "
                            f"= {max_input_len + requested} exceeds model context ({self.max_model_len}). "
                            f"Reduce input length or max_tokens."
                        )
                        state.finished = True
                    force_cleanup = True
                    return
                requested = remaining

            generation_kwargs["max_new_tokens"] = requested

            def _generate() -> dict[str, Any]:
                result: dict[str, Any] = {}
                try:
                    with torch.inference_mode():
                        result["output"] = self.model.generate(**generation_kwargs)
                except Exception as exc:
                    result["error"] = exc
                return result

            result = await asyncio.to_thread(_generate)

            if "error" in result:
                error_msg = str(result["error"])
                force_cleanup = True
                for state in batch:
                    state.error = error_msg
                    state.finished = True
                return

            output_ids = result["output"]
            input_len = input_ids.shape[1]
            inputs = None
            generation_kwargs = None

            for i, state in enumerate(batch):
                generated_ids = output_ids[i][input_len:].detach().cpu()
                # Strip padding from generated tokens
                if self.processor.tokenizer.pad_token_id is not None:
                    generated_ids = generated_ids[
                        generated_ids != self.processor.tokenizer.pad_token_id
                    ]
                # Strip eos
                if self.processor.tokenizer.eos_token_id is not None:
                    eos_id = self.processor.tokenizer.eos_token_id
                    if isinstance(eos_id, list):
                        for eid in eos_id:
                            generated_ids = generated_ids[generated_ids != eid]
                    else:
                        generated_ids = generated_ids[generated_ids != eos_id]

                state.generated_text = self.processor.tokenizer.decode(
                    generated_ids, skip_special_tokens=True,
                )
                state.completion_token_count = len(generated_ids)
                state.finished = True
        finally:
            self._release_batch_state(batch)
            del inputs
            del input_ids
            del generation_kwargs
            del output_ids
            await self._maybe_cleanup_memory(force=force_cleanup)
