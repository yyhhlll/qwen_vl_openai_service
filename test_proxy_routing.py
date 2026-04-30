from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


TEXT_URL = "http://text.local"
VISION_URL = "http://vision.local"


class FakeHTTPXRequest:
    def __init__(self, method: str, url: str) -> None:
        self.method = method
        self.url = url


class FakeHTTPXResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_payload: Any | None = None,
        text: str = "",
        request: Any | None = None,
    ) -> None:
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text or (json.dumps(json_payload, ensure_ascii=False) if json_payload is not None else "")
        self.request = request
        self.headers = {"content-type": "application/json"} if json_payload is not None else {}

    def json(self) -> Any:
        if self._json_payload is None:
            raise ValueError("no json")
        return self._json_payload


class FakeJSONResponse:
    def __init__(self, content: Any, status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.body = json.dumps(content, ensure_ascii=False).encode("utf-8")


class FakeStreamingResponse:
    def __init__(self, iterator, media_type: str | None = None, headers: dict[str, str] | None = None) -> None:
        self.iterator = iterator
        self.media_type = media_type
        self.headers = headers or {}


class FakeHTTPException(Exception):
    def __init__(self, status_code: int, detail: Any) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakeChatCompletionRequest:
    def __init__(self, model: str, messages: list[Any], stream: bool = False) -> None:
        self.model = model
        self.messages = messages
        self.stream = stream

    @classmethod
    def model_validate(cls, body: dict[str, Any]):
        if "model" not in body or "messages" not in body:
            raise FakeValidationError("missing fields")
        return cls(body["model"], body["messages"], bool(body.get("stream", False)))


class FakeValidationError(Exception):
    def errors(self):
        return [{"msg": str(self)}]


def install_import_stubs() -> None:
    fake_httpx = types.ModuleType("httpx")
    fake_httpx.Timeout = lambda **_: object()
    fake_httpx.Request = FakeHTTPXRequest
    fake_httpx.Response = FakeHTTPXResponse
    fake_httpx.AsyncClient = lambda timeout=None: object()

    fake_fastapi = types.ModuleType("fastapi")

    class FakeFastAPI:
        def get(self, _path):
            return lambda func: func

        def post(self, _path):
            return lambda func: func

    fake_fastapi.FastAPI = FakeFastAPI
    fake_fastapi.Header = lambda default=None: default
    fake_fastapi.HTTPException = FakeHTTPException
    fake_fastapi.Request = object
    fake_fastapi.Response = object

    fake_responses = types.ModuleType("fastapi.responses")
    fake_responses.JSONResponse = FakeJSONResponse
    fake_responses.StreamingResponse = FakeStreamingResponse

    fake_pydantic = types.ModuleType("pydantic")
    fake_pydantic.ValidationError = FakeValidationError

    fake_protocol = types.ModuleType("protocol")
    fake_protocol.ChatCompletionRequest = FakeChatCompletionRequest

    sys.modules.update(
        {
            "httpx": fake_httpx,
            "fastapi": fake_fastapi,
            "fastapi.responses": fake_responses,
            "pydantic": fake_pydantic,
            "protocol": fake_protocol,
        }
    )


class FakeRequest:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    async def json(self) -> dict[str, Any]:
        return self._body

    async def is_disconnected(self) -> bool:
        return False


class FakeStreamResponse:
    def __init__(self, chunks: list[bytes] | None = None) -> None:
        self.chunks = chunks or [b"data: ok\n\n"]
        self.closed = False

    async def aiter_bytes(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, responses: dict[str, list[FakeHTTPXResponse]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    async def get(self, url: str) -> FakeHTTPXResponse:
        return FakeHTTPXResponse(200, {"ok": True}, request=FakeHTTPXRequest("GET", url))

    async def post(
        self,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> FakeHTTPXResponse:
        self.calls.append({"url": url, "headers": headers, "json": json})
        key = "text" if url.startswith(TEXT_URL) else "vision"
        queue = self.responses.setdefault(key, [])
        if not queue:
            raise AssertionError(f"No fake response queued for {key}")
        return queue.pop(0)

    def build_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json: dict[str, Any],
    ) -> dict[str, Any]:
        request = {"method": method, "url": url, "headers": headers, "json": json}
        self.stream_calls.append(request)
        return request

    async def send(self, request: dict[str, Any], stream: bool = False) -> FakeStreamResponse:
        return FakeStreamResponse()


class FailingSendClient(FakeClient):
    async def send(self, request: dict[str, Any], stream: bool = False) -> FakeStreamResponse:
        raise RuntimeError("stream send failed")


def chat_response(content: str, model: str = "backend-model", status: int = 200) -> FakeHTTPXResponse:
    return FakeHTTPXResponse(
        status,
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        request=FakeHTTPXRequest("POST", "http://fake.local/v1/chat/completions"),
    )


def load_proxy():
    install_import_stubs()
    env = {
        "API_KEY": "1234",
        "TEXT_BACKEND_URLS": TEXT_URL,
        "BACKEND_URLS": VISION_URL,
        "TEXT_MODEL_NAME": "Qwen3.5-0.6B",
        "VISION_MODEL_NAME": "Qwen3.5-4B",
    }
    with patch.dict(os.environ, env, clear=False):
        sys.modules.pop("proxy", None)
        return importlib.import_module("proxy")


class ProxyRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.proxy = load_proxy()

    async def call_chat(self, body: dict[str, Any], client: FakeClient):
        self.proxy._client = client
        return await self.proxy.chat_completions(FakeRequest(body), authorization="Bearer 1234")

    def response_json(self, response) -> dict[str, Any]:
        return json.loads(response.body.decode("utf-8"))

    async def test_text_compliant_uses_only_text_pool_and_preserves_model(self) -> None:
        client = FakeClient({"text": [chat_response('{"compliant": true}', model="Qwen3.5-0.6B")]})
        body = {"model": "external-compliance", "messages": [{"role": "user", "content": "hello"}], "stream": False}
        response = await self.call_chat(body, client)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["url"].startswith(TEXT_URL))
        self.assertEqual(client.calls[0]["json"]["model"], "Qwen3.5-0.6B")
        payload = self.response_json(response)
        self.assertEqual(payload["model"], "external-compliance")
        self.assertEqual(payload["choices"][0]["message"]["content"], '{"compliant": true}')
        self.assertEqual(response.headers["X-Qwen-Route"], "text_0_6b_only")
        self.assertEqual(response.headers["X-Qwen-Text-Backend"], TEXT_URL)

    async def test_text_noncompliant_escalates_to_vision_pool(self) -> None:
        client = FakeClient({"text": [chat_response('{"compliant": false}')], "vision": [chat_response('{"category": "risk"}')]} )
        body = {"model": "external-compliance", "messages": [{"role": "user", "content": "bad"}], "stream": False}
        response = await self.call_chat(body, client)
        self.assertEqual([call["url"].split("/")[2] for call in client.calls], ["text.local", "vision.local"])
        self.assertEqual(client.calls[0]["json"]["model"], "Qwen3.5-0.6B")
        self.assertEqual(client.calls[1]["json"]["model"], "Qwen3.5-4B")
        self.assertEqual(self.response_json(response)["choices"][0]["message"]["content"], '{"category": "risk"}')
        self.assertEqual(response.headers["X-Qwen-Route"], "text_0_6b_then_4b")
        self.assertEqual(response.headers["X-Qwen-Text-Backend"], TEXT_URL)
        self.assertEqual(response.headers["X-Qwen-Vision-Backend"], VISION_URL)

    async def test_text_non_2xx_escalates_to_vision_pool(self) -> None:
        client = FakeClient(
            {
                "text": [FakeHTTPXResponse(503, {"error": "busy"})],
                "vision": [chat_response("4b fallback")],
            }
        )
        body = {"model": "external-compliance", "messages": [{"role": "user", "content": "maybe"}], "stream": False}
        response = await self.call_chat(body, client)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(client.calls[1]["url"].startswith(VISION_URL))
        self.assertEqual(self.response_json(response)["choices"][0]["message"]["content"], "4b fallback")

    async def test_image_request_uses_only_vision_pool(self) -> None:
        client = FakeClient({"vision": [chat_response("vision result")]})
        body = {
            "model": "external-compliance",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "check"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ],
            "stream": False,
        }
        response = await self.call_chat(body, client)
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(client.calls[0]["url"].startswith(VISION_URL))
        self.assertEqual(self.response_json(response)["choices"][0]["message"]["content"], "vision result")
        self.assertEqual(response.headers["X-Qwen-Route"], "image_direct_4b")
        self.assertEqual(response.headers["X-Qwen-Vision-Backend"], VISION_URL)

    async def test_streaming_text_request_uses_only_vision_pool(self) -> None:
        client = FakeClient()
        body = {
            "model": "external-compliance",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        }
        response = await self.call_chat(body, client)
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(len(client.stream_calls), 1)
        self.assertTrue(client.stream_calls[0]["url"].startswith(VISION_URL))
        self.assertEqual(client.stream_calls[0]["json"]["model"], "Qwen3.5-4B")
        self.assertEqual(response.media_type, "text/event-stream")

    async def test_streaming_send_failure_releases_inflight(self) -> None:
        client = FailingSendClient()
        body = {
            "model": "external-compliance",
            "messages": [{"role": "user", "content": "stream"}],
            "stream": True,
        }

        with self.assertRaises(RuntimeError):
            await self.call_chat(body, client)

        self.assertEqual(self.proxy.VISION_POOL.inflight(VISION_URL), 0)

    async def test_streaming_image_request_uses_only_vision_pool(self) -> None:
        client = FakeClient()
        body = {
            "model": "external-compliance",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": "http://image"}}],
                }
            ],
            "stream": True,
        }
        await self.call_chat(body, client)
        self.assertEqual(len(client.calls), 0)
        self.assertEqual(len(client.stream_calls), 1)
        self.assertTrue(client.stream_calls[0]["url"].startswith(VISION_URL))

    async def test_text_backend_receives_minimal_0_6b_safe_body(self) -> None:
        client = FakeClient({"text": [chat_response('{"compliant": true}')]})
        body = {
            "model": "external-compliance",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
            "temperature": 0,
            "top_p": 0.8,
            "response_format": {"type": "json_object"},
            "stop": ["END"],
            "seed": 7,
            "presence_penalty": 0.5,
            "max_tokens": 64,
        }
        await self.call_chat(body, client)
        upstream = client.calls[0]["json"]
        self.assertEqual(
            upstream,
            {
                "model": "Qwen3.5-0.6B",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            },
        )

    async def test_text_only_content_parts_are_flattened_for_0_6b(self) -> None:
        client = FakeClient({"text": [chat_response('{"compliant": true}')]})
        body = {
            "model": "external-compliance",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": "world"},
                    ],
                }
            ],
            "stream": False,
        }
        await self.call_chat(body, client)
        upstream = client.calls[0]["json"]
        self.assertEqual(upstream["messages"], [{"role": "user", "content": "hello\nworld"}])

    async def test_vision_fallback_preserves_original_options_except_model(self) -> None:
        client = FakeClient({"text": [chat_response('{"compliant": false}')], "vision": [chat_response('{"category":"risk"}')]})
        body = {
            "model": "external-compliance",
            "messages": [{"role": "user", "content": "bad"}],
            "stream": False,
            "temperature": 0.2,
            "top_p": 0.8,
            "response_format": {"type": "json_object"},
            "stop": ["END"],
            "seed": 7,
            "presence_penalty": 0.5,
            "max_tokens": 64,
        }
        await self.call_chat(body, client)
        upstream = client.calls[1]["json"]
        self.assertEqual(upstream["model"], "Qwen3.5-4B")
        self.assertEqual(upstream["stream"], False)
        self.assertEqual(upstream["temperature"], 0.2)
        self.assertEqual(upstream["top_p"], 0.8)
        self.assertEqual(upstream["response_format"], {"type": "json_object"})
        self.assertEqual(upstream["stop"], ["END"])
        self.assertEqual(upstream["seed"], 7)
        self.assertEqual(upstream["presence_penalty"], 0.5)

    async def test_models_returns_deterministic_merged_list(self) -> None:
        response = await self.proxy.list_models(authorization="Bearer 1234")
        payload = self.response_json(response)
        self.assertEqual(payload["object"], "list")
        self.assertEqual([item["id"] for item in payload["data"]], ["Qwen3.5-0.6B", "Qwen3.5-4B"])


class ProxyParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = load_proxy()

    def payload(self, content: Any) -> dict[str, Any]:
        return {"object": "chat.completion", "choices": [{"message": {"content": content}}]}

    def test_compliant_fixtures(self) -> None:
        for value in [
            '{"compliant": true}',
            '{"is_compliant": true}',
            '{"result": "合规"}',
            '{"status": "compliant"}',
            "合规",
            "compliant",
        ]:
            with self.subTest(value=value):
                self.assertTrue(self.proxy.is_confidently_compliant(self.payload(value)))


    def test_qwen_guard_safe_text_is_compliant(self) -> None:
        self.assertTrue(self.proxy.is_confidently_compliant(self.payload("Safety: Safe\nCategories: None")))

    def test_markdown_json_safe_text_is_compliant(self) -> None:
        self.assertTrue(self.proxy.is_confidently_compliant(self.payload('```json\n{"compliant": true, "category": []}\n```')))

    def test_legacy_is_safty_key_is_compliant(self) -> None:
        self.assertTrue(self.proxy.is_confidently_compliant(self.payload('{"is_safty": true, "unsafy_class": []}')))

    def test_noncompliant_and_ambiguous_fixtures(self) -> None:
        for value in [
            '{"compliant": false}',
            '{"result": "不合规"}',
            "不合规",
            "not compliant",
            "{bad json",
            "合规 but also 不合规",
            "",
            '{"category": "risk"}',
        ]:
            with self.subTest(value=value):
                self.assertFalse(self.proxy.is_confidently_compliant(self.payload(value)))

    def test_missing_choices_or_content_is_not_compliant(self) -> None:
        for payload in (
            {},
            {"choices": []},
            {"choices": [{"message": {}}]},
            {"choices": [{"message": {"content": None}}]},
        ):
            with self.subTest(payload=payload):
                self.assertFalse(self.proxy.is_confidently_compliant(payload))


class DeploymentTopologyTests(unittest.TestCase):
    def test_root_compose_uses_expected_4b_backend_sets(self) -> None:
        compose = Path("docker-compose.yml").read_text()
        for item in [f"10.2.0.129:800{i}" for i in range(1, 5)]:
            self.assertNotIn(item, compose)
        expected = [
            *(f"10.2.0.129:800{i}" for i in range(5, 9)),
            *(f"10.2.0.130:800{i}" for i in range(1, 9)),
            *(f"10.2.0.131:800{i}" for i in range(1, 4)),
        ]
        for item in expected:
            self.assertIn(item, compose)
        for service in [f"qwen35-129-g{i}:" for i in range(4)]:
            self.assertNotIn(service, compose)

    def test_total_nginx_routes_only_to_proxy_gateway(self) -> None:
        nginx = Path("qwen35.nginx.conf").read_text()
        self.assertIn("upstream qwen35_proxy_gateway", nginx)
        self.assertIn("server 127.0.0.1:19000", nginx)
        self.assertIn("proxy_pass http://qwen35_proxy_gateway", nginx)
        self.assertNotIn("upstream qwen35_backend", nginx)
        for port in [
            *(f"10.2.0.129:800{i}" for i in range(1, 9)),
            *(f"10.2.0.130:800{i}" for i in range(1, 9)),
            *(f"10.2.0.131:800{i}" for i in range(1, 9)),
        ]:
            self.assertNotIn(port, nginx)

    def test_text_configs_use_only_129_g0_to_g3(self) -> None:
        compose = Path("qwen_text_vllm_service/docker-compose.yml").read_text()
        nginx = Path("qwen_text_vllm_service/qwen35_text.nginx.conf").read_text()
        for item in [f"10.2.0.129:1200{i}" for i in range(1, 5)]:
            self.assertIn(item, nginx)
        forbidden = [
            *(f"10.2.0.129:1200{i}" for i in range(5, 9)),
            *(f"10.2.0.130:1200{i}" for i in range(1, 9)),
            *(f"10.2.0.131:1200{i}" for i in range(1, 4)),
        ]
        for item in forbidden:
            self.assertNotIn(item, nginx)
        forbidden_services = [
            *(f"qwen35-text-129-g{i}:" for i in range(4, 8)),
            *(f"qwen35-text-130-g{i}:" for i in range(8)),
            *(f"qwen35-text-131-g{i}:" for i in range(4, 7)),
        ]
        for service in forbidden_services:
            self.assertNotIn(service, compose)


class RenderedComposeTopologyTests(unittest.TestCase):
    def compose_services(self, cwd: str, profile: str) -> set[str]:
        try:
            result = subprocess.run(
                ["docker", "compose", "--profile", profile, "config", "--services"],
                cwd=cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError:
            self.skipTest("docker is not installed")
        except subprocess.CalledProcessError as exc:
            self.fail(f"docker compose config failed: {exc.stderr}")
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def test_rendered_root_host_profiles_match_4b_topology(self) -> None:
        root = "."
        self.assertEqual(
            self.compose_services(root, "host129"),
            {
                "qwen35-129-g4",
                "qwen35-129-g5",
                "qwen35-129-g6",
                "qwen35-129-g7",
                "qwen35-proxy",
            },
        )
        self.assertEqual(
            self.compose_services(root, "host130"),
            {f"qwen35-130-g{i}" for i in range(8)},
        )
        self.assertEqual(
            self.compose_services(root, "host131"),
            {"qwen35-131-g4", "qwen35-131-g5", "qwen35-131-g6"},
        )

    def test_rendered_text_profiles_match_06b_topology(self) -> None:
        text_dir = "qwen_text_vllm_service"
        self.assertEqual(
            self.compose_services(text_dir, "host129"),
            {
                "qwen35-text-129-g0",
                "qwen35-text-129-g1",
                "qwen35-text-129-g2",
                "qwen35-text-129-g3",
            },
        )
        self.assertEqual(self.compose_services(text_dir, "host130"), set())
        self.assertEqual(self.compose_services(text_dir, "host131"), set())



if __name__ == "__main__":
    unittest.main()
