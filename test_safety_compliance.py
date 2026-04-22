#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test a safety-compliance prompt against the local OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="Chat completions endpoint URL.",
    )
    parser.add_argument(
        "--api-key",
        default="1234",
        help="Bearer API key.",
    )
    parser.add_argument(
        "--model",
        default="Qwen3.5-27B-VL",
        help="Model name passed to the endpoint.",
    )
    parser.add_argument(
        "--system-prompt-file",
        help="Path to a UTF-8 text file that contains the full system prompt.",
    )
    parser.add_argument(
        "--payload-file",
        help="Path to a JSON file containing the full request payload to send as-is.",
    )
    parser.add_argument(
        "--user-input",
        help="Raw user input that will be wrapped as the content to analyze.",
    )
    parser.add_argument(
        "--image-file",
        help="Local image file to include as a data URL in the user message.",
    )
    parser.add_argument(
        "--user-text",
        help="Raw user input that will be wrapped as the content to analyze.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Request timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.payload_file:
        payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    else:
        if not args.system_prompt_file:
            raise SystemExit("--system-prompt-file is required unless --payload-file is used.")
        if not args.user_input and not args.image_file and not args.user_text:
            raise SystemExit("At least one of --user-input, --user-text, or --image-file is required.")

        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

        if args.image_file:
            user_content: list[dict[str, object]] = []
            user_text = args.user_text or "待分析用户输入"
            if user_text:
                user_content.append({"type": "text", "text": user_text})

            image_path = Path(args.image_file)
            mime_type, _ = mimetypes.guess_type(image_path.name)
            if not mime_type:
                mime_type = "image/png"
            image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                }
            )
        else:
            text_value = args.user_input if args.user_input is not None else (args.user_text or "")
            user_content = f"待分析用户输入：\n```text\n{text_value}\n```"

        payload = {
            "model": args.model,
            "stream": False,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    started_at = time.time()
    request = Request(
        args.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"status: {exc.code}")
        print(f"elapsed: {time.time() - started_at:.3f} s")
        print("error:")
        print(error_body)
        raise
    except URLError as exc:
        print(f"elapsed: {time.time() - started_at:.3f} s")
        print(f"request_failed: {exc}")
        raise

    ended_at = time.time()
    print(f"status: {status}")
    print(f"elapsed: {ended_at - started_at:.3f} s")

    data = json.loads(body)

    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)
    tps = completion_tokens / (ended_at - started_at) if ended_at > started_at else 0.0

    print(f"prompt_tokens: {prompt_tokens}")
    print(f"completion_tokens: {completion_tokens}")
    print(f"total_tokens: {total_tokens}")
    print(f"tokens_per_second: {tps:.3f}")
    print("content:")
    print(data["choices"][0]["message"]["content"])

    # If the model followed instructions, this should itself be valid JSON.
    try:
        parsed = json.loads(data["choices"][0]["message"]["content"])
    except json.JSONDecodeError:
        return

    print("parsed_json:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
