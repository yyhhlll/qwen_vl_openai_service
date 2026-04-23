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


def _load_scheduler_module():
    fake_engine = types.ModuleType("engine")
    fake_engine.TransformersVLEngine = object

    fake_state = types.ModuleType("state")
    fake_state.RequestState = object

    with patch.dict(
        sys.modules,
        {
            "engine": fake_engine,
            "state": fake_state,
        },
    ):
        if "scheduler" in sys.modules:
            del sys.modules["scheduler"]
        return importlib.import_module("scheduler")


class _FakeConvertedImage:
    def __init__(self) -> None:
        self.convert_calls: list[str] = []

    def convert(self, mode: str):
        self.convert_calls.append(mode)
        return {"mode": mode}


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class ImageUrlSupportTests(unittest.TestCase):
    def test_remote_image_url_is_blocked_when_disabled(self) -> None:
        engine = _load_engine_module()
        fake_image_module = types.SimpleNamespace(open=lambda _: _FakeConvertedImage())
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = fake_image_module

        test_engine = engine.TransformersVLEngine(
            "unused",
            allow_remote_image_urls=False,
        )

        with patch.dict("sys.modules", {"PIL": fake_pil}):
            with self.assertRaisesRegex(
                ValueError,
                "Remote image URLs are disabled by configuration",
            ):
                test_engine._load_image("http://10.2.0.129:9000/example.png")

    def test_remote_image_url_is_loaded_when_enabled(self) -> None:
        engine = _load_engine_module()
        opened_payloads: list[bytes] = []

        def fake_open(buffer) -> _FakeConvertedImage:
            opened_payloads.append(buffer.read())
            return _FakeConvertedImage()

        fake_image_module = types.SimpleNamespace(open=fake_open)
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = fake_image_module

        test_engine = engine.TransformersVLEngine(
            "unused",
            allow_remote_image_urls=True,
        )

        with patch.dict("sys.modules", {"PIL": fake_pil}):
            with patch.object(
                engine,
                "urlopen",
                return_value=_FakeResponse(b"remote-image-bytes"),
            ) as mocked_urlopen:
                loaded = test_engine._load_image(
                    "http://10.2.0.129:9000/llm-detect/example.png"
                )

        mocked_urlopen.assert_called_once_with(
            "http://10.2.0.129:9000/llm-detect/example.png"
        )
        self.assertEqual(opened_payloads, [b"remote-image-bytes"])
        self.assertEqual(loaded, {"mode": "RGB"})


class SchedulerBatchSplitTests(unittest.TestCase):
    def test_scheduler_splits_text_and_image_requests(self) -> None:
        scheduler_module = _load_scheduler_module()
        scheduler = scheduler_module.Scheduler(engine=object())

        text_a = types.SimpleNamespace(images=[], name="text-a")
        image_a = types.SimpleNamespace(images=[object()], name="image-a")
        text_b = types.SimpleNamespace(images=[], name="text-b")
        image_b = types.SimpleNamespace(images=[object()], name="image-b")

        split = scheduler._split_batch_by_image_modality(
            [text_a, image_a, text_b, image_b]
        )

        self.assertEqual(split, [[text_a, text_b], [image_a, image_b]])


if __name__ == "__main__":
    unittest.main()
