from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _load_engine_module():
    fake_protocol = types.ModuleType("protocol")
    fake_protocol.ChatMessage = object

    fake_state = types.ModuleType("state")
    fake_state.RequestState = object

    with patch.dict(
        sys.modules,
        {
            "protocol": fake_protocol,
            "state": fake_state,
        },
    ):
        if "engine" in sys.modules:
            del sys.modules["engine"]
        return importlib.import_module("engine")


class MemoryCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_cleanup_runs_only_after_interval(self) -> None:
        engine = _load_engine_module()
        cleanup_calls: list[str] = []
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: cleanup_calls.append("empty_cache"),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda

        test_engine = engine.TransformersVLEngine(
            "unused",
            memory_cleanup_interval=2,
        )

        with patch.object(engine.gc, "collect", side_effect=lambda: cleanup_calls.append("gc")):
            with patch.dict("sys.modules", {"torch": fake_torch}):
                await test_engine._maybe_cleanup_memory()
                self.assertEqual(cleanup_calls, [])

                await test_engine._maybe_cleanup_memory()
                self.assertEqual(cleanup_calls, ["gc", "empty_cache"])

    async def test_force_cleanup_works_when_interval_disabled(self) -> None:
        engine = _load_engine_module()
        cleanup_calls: list[str] = []
        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: cleanup_calls.append("empty_cache"),
        )
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = fake_cuda

        test_engine = engine.TransformersVLEngine(
            "unused",
            memory_cleanup_interval=0,
        )

        with patch.object(engine.gc, "collect", side_effect=lambda: cleanup_calls.append("gc")):
            with patch.dict("sys.modules", {"torch": fake_torch}):
                await test_engine._maybe_cleanup_memory(force=True)

        self.assertEqual(cleanup_calls, ["gc", "empty_cache"])

    def test_release_batch_state_drops_images_and_prompt_text(self) -> None:
        engine = _load_engine_module()
        batch = [
            types.SimpleNamespace(images=[object(), object()], prompt_text="prompt-a"),
            types.SimpleNamespace(images=[object()], prompt_text="prompt-b"),
        ]

        test_engine = engine.TransformersVLEngine("unused")
        test_engine._release_batch_state(batch)

        self.assertEqual(batch[0].images, [])
        self.assertEqual(batch[1].images, [])
        self.assertEqual(batch[0].prompt_text, "")
        self.assertEqual(batch[1].prompt_text, "")


if __name__ == "__main__":
    unittest.main()
