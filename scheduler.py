from __future__ import annotations

import asyncio

try:
    from .engine import TransformersVLEngine
    from .state import RequestState
except ImportError:
    from engine import TransformersVLEngine
    from state import RequestState


class Scheduler:
    """Batch scheduler: collects requests over a time window, then runs batch inference."""

    def __init__(
        self,
        engine: TransformersVLEngine,
        max_batch_size: int = 4,
        batch_wait_ms: float = 50.0,
    ) -> None:
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_wait_ms = batch_wait_ms
        self._queue: asyncio.Queue[RequestState] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def submit(self, state: RequestState) -> None:
        await self._queue.put(state)

    async def _collect_batch(self) -> list[RequestState]:
        """Wait for first request, then collect more within the time window."""
        first = await self._queue.get()
        batch = [first]

        deadline = asyncio.get_event_loop().time() + self.batch_wait_ms / 1000.0
        while len(batch) < self.max_batch_size:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                break
            try:
                state = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(state)
            except asyncio.TimeoutError:
                break
        return batch

    async def _loop(self) -> None:
        while True:
            batch = await self._collect_batch()
            try:
                await self.engine.generate_batch(batch)
            except Exception as exc:
                for state in batch:
                    state.error = str(exc)
                    state.finished = True
            finally:
                for state in batch:
                    if not state.result_future.done():
                        state.result_future.set_result(None)
