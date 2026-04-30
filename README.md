# qwen_vl_openai_service

一个基于 `transformers` + `FastAPI` 的本地 OpenAI 兼容 Qwen-VL 服务，适合在没有 vLLM 路径、但仍需要稳定提供 `/v1/chat/completions` 接口时使用。

当前仓库包含两种运行形态：

- `server.py`：单个 Qwen-VL 后端实例
- `proxy.py`：把多个后端实例聚合成一个统一入口

部署层也支持一个额外的可选入口：

- `qwen35.nginx.conf`：总入口 nginx，转发到 `qwen35-proxy:19000`

## 当前能力

- 提供 `/health`
- 提供 `/v1/models`
- 提供 `/v1/chat/completions`
- 支持 OpenAI 风格的 `messages`
- 支持纯文本和图文混合输入
- 支持三种图片来源：
  - 本地文件路径
  - `http(s)` 图片 URL
  - `data:image/...` base64 URL
- 支持 Bearer Token 鉴权
- 支持基于短时间窗口的 batch-size 批量推理
- 支持代理层按“当前 in-flight 最少”选择后端
- 返回 `usage.prompt_tokens` / `completion_tokens` / `total_tokens`
- 请求里的 `model` 字段不会被强校验，响应中会原样回传，方便上层网关或评测工具传别名

## 重要限制

- `stream=true` 不是逐 token 实时流式推理，而是“完整生成结束后”再模拟成 SSE 分片输出。
- 这是单进程 `transformers.generate(...)` 服务，不是 vLLM，也不提供 token 级连续批处理。
- 调度器会把纯文本请求和含图请求拆成不同子批次，避免同一批次里混入有图/无图请求。
- 模型默认可按首次请求懒加载；Docker Compose 部署默认 `LOAD_MODEL_ON_STARTUP=1`，容器启动时加载模型。

## 仓库结构

- [server.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/server.py): 单实例 OpenAI 兼容服务
- [engine.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/engine.py): 模型加载、消息归一化、图片读取、批量生成
- [scheduler.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/scheduler.py): 简单批量调度器
- [proxy.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/proxy.py): 多后端聚合代理
- [protocol.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/protocol.py): OpenAI 兼容请求/响应协议模型
- [state.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/state.py): 单请求运行状态
- [test_safety_compliance.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/test_safety_compliance.py): 本地安全审查/合规提示词测试脚本
- [qwen35.nginx.conf](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/qwen35.nginx.conf): 总入口 `nginx` 配置，转发到 `qwen35-proxy:19000`
  - 已针对大模型长上下文请求做了入口调优：增大 `client_body_buffer_size`，并关闭 `proxy_request_buffering`，避免大请求体频繁落盘后再转发

## 依赖

仓库里没有锁定依赖文件，按源码至少需要这些包：

```bash
pip install fastapi uvicorn "pydantic>=2" httpx pillow transformers torch
```

说明：

- `torch`、`transformers` 的具体版本需要和你的模型、CUDA/HIP 环境匹配。
- 如果使用本地图片路径，路径必须对后端进程可见。
- 当前部署默认是离线模式：
  - 模型和 processor 必须已经在本地模型目录里
  - 默认允许远程图片 URL；如需完全离线输入，可设置 `ALLOW_REMOTE_IMAGE_URLS=0`

## 环境变量

后端 `server.py` 使用这些环境变量：

- `MODEL_NAME`
  - 默认值：`Qwen3.5-27B-VL`
  - `/v1/models` 返回的模型名
- `MODEL_PATH`
  - 默认值：`/models/Qwen3.5-27B`
  - Hugging Face 模型目录或本地模型路径
- `API_KEY`
  - 默认值：`1234`
  - 为空时不校验鉴权
- `DEVICE_MAP`
  - 默认值：`cuda`
  - 强制模型加载到当前容器可见的 DCU/GPU；如需恢复自动放置，可显式设为 `auto`
- `HIP_VISIBLE_DEVICES`
  - Docker Compose 部署用于给每个后端实例绑定单张 DCU/GPU
  - 当前镜像不要同时设置 `ROCR_VISIBLE_DEVICES`，否则 torch 可能无法正确初始化 HIP
- `MAX_BATCH_SIZE`
  - 默认值：`4`
  - 每批最多处理多少请求
- `MAX_RUNNING`
  - `MAX_BATCH_SIZE` 的兼容别名
- `BATCH_WAIT_MS`
  - 默认值：`50`
  - 收集一批请求的等待窗口
- `MAX_MODEL_LEN`
  - 默认值：`0`
  - 为 `0` 时自动使用模型配置里的上下文长度
- `GPU_MEMORY_CLEANUP_INTERVAL`
  - 默认值：`32`
  - 每处理多少个 batch 触发一次 `gc.collect()` + `torch.cuda.empty_cache()`
  - 设为 `0` 表示关闭周期性清理
- `OFFLINE_MODE`
  - 默认值：`1`
  - 离线加载本地模型和 processor，不访问远程模型仓库
- `ALLOW_REMOTE_IMAGE_URLS`
  - 默认值：`1`
  - 默认允许 `http(s)` 图片 URL
  - 仍然支持本地图片路径和 `data:image/...` base64
  - 如需完全离线或禁用远程取图，可显式设为 `0`
- `REMOTE_IMAGE_TIMEOUT_SECONDS`
  - 默认值：`60`
  - 远程图片抓取超时时间（秒）
  - 用于避免内网图片源卡住时把整个请求长时间挂死

代理 `proxy.py` 额外使用：

- `BACKEND_URLS`
  - 4B 多模态后端地址列表；生产默认由 `docker-compose.yml` 设置为 `129:8005-8008 + 130:8001-8008 + 131:8001-8003`
  - 兼容旧变量名；也可用 `VISION_BACKEND_URLS` / `BACKEND_URLS_4B`
- `TEXT_BACKEND_URLS`
  - 0.6B 文本后端地址列表；生产默认 `http://10.2.0.129:12001,http://10.2.0.129:12002,http://10.2.0.129:12003,http://10.2.0.129:12004`
- `TEXT_MODEL_NAME` / `VISION_MODEL_NAME`
  - 内部转发到不同后端时会重写 `model` 字段；非流式响应尽量恢复调用方原始 `model`
  - 转发到 0.6B 文本池时使用兼容最小请求体：`model`、`messages`、`max_tokens`；会剥离 `response_format`、`top_p`、`stop`、`seed`、`presence_penalty`、显式 `stream=false` 等容易导致 0.6B vLLM 报错的参数。若 `messages[].content` 是纯文本 part 列表，会压平成字符串。
  - 升级到 4B 或图片直达 4B 时保留原始请求参数，仅改写内部 `model`。

## 启动单后端

单实例 batch 推理启动方式：

```bash
MODEL_PATH=/models/Qwen3.5-27B \
MODEL_NAME=Qwen3.5-27B-VL \
API_KEY=1234 \
DEVICE_MAP=cuda \
MAX_BATCH_SIZE=4 \
BATCH_WAIT_MS=50 \
GPU_MEMORY_CLEANUP_INTERVAL=32 \
OFFLINE_MODE=1 \
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

如果你在多卡机器上跑一个跨多卡副本，可以额外设置设备环境变量，例如：

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
MODEL_PATH=/models/Qwen3.5-27B \
MODEL_NAME=Qwen3.5-27B-VL \
API_KEY=1234 \
DEVICE_MAP=cuda \
MAX_BATCH_SIZE=4 \
BATCH_WAIT_MS=50 \
GPU_MEMORY_CLEANUP_INTERVAL=32 \
OFFLINE_MODE=1 \
python3 -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## 启动统一代理

先分别启动 4B 多模态后端和 0.6B 文本后端，例如在 `129` 本机：

- 4B：`http://127.0.0.1:8005` ... `http://127.0.0.1:8008`
- 0.6B：`http://127.0.0.1:12001` ... `http://127.0.0.1:12004`

然后启动代理：

```bash
API_KEY=1234 \
BACKEND_URLS=http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007,http://127.0.0.1:8008 \
TEXT_BACKEND_URLS=http://127.0.0.1:12001,http://127.0.0.1:12002,http://127.0.0.1:12003,http://127.0.0.1:12004 \
python3 -m uvicorn proxy:app --host 0.0.0.0 --port 8000
```

代理会：

- 暴露同样的 `/health`、`/v1/models`、`/v1/chat/completions`
- 维护 0.6B 文本池和 4B 多模态池的 in-flight 计数
- 文本非流式请求先走 0.6B；明确合规则直接返回，否则升级 4B
- 图片/多模态请求和所有 `stream=true` 请求直接走 4B
- 透明转发 SSE 流式响应；流式 chunk 的 `model` 字段不做归一化保证

## 接口示例

### 1. 查看模型列表

```bash
curl http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer 1234"
```

### 2. 纯文本请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{
    "model": "any-alias-is-accepted",
    "stream": false,
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "请用一句话介绍这项服务。"}
    ]
  }'
```

### 3. 带图片请求

本地图片路径示例：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -d '{
    "model": "Qwen3.5-27B-VL",
    "stream": false,
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请描述这张图片。"},
          {"type": "image_url", "image_url": {"url": "/absolute/path/to/example.png"}}
        ]
      }
    ]
  }'
```

也可以把 `url` 换成：

- `data:image/png;base64,...`

也可以直接传服务端能访问到的远程图片 URL，例如内网地址 `http://10.2.0.129:9000/...`。
如需完全离线运行，可把 `ALLOW_REMOTE_IMAGE_URLS=0` 关掉远程取图。
如果远程图片源偶发卡顿，可调小 `REMOTE_IMAGE_TIMEOUT_SECONDS`，让失败尽快返回而不是长时间超时。

### 4. SSE 伪流式输出

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer 1234" \
  -N \
  -d '{
    "model": "Qwen3.5-27B-VL",
    "stream": true,
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "请简短介绍你自己。"}
    ]
  }'
```

注意：这里的 SSE 仅仅是把完整输出拆成多个 chunk 返回，不代表模型正在实时逐 token 解码。

## `/health` 返回内容

单后端返回：

- `ok`
- `model`
- `max_model_len`
- `binding.device_map`
- `binding.hip_visible_devices`
- `binding.rocr_visible_devices`
- `binding.model_device`
- `scheduler.queue_size`
- `scheduler.max_batch_size`
- `scheduler.batch_wait_ms`

代理返回：

- `ok`
- `backends[]`
- 每个后端的 `status_code`
- 每个后端的 `proxy_inflight`
- 后端自己的 `/health` 内容或错误信息

## 生成参数行为

当前批处理的生成参数取自该批次第一个请求，因此更适合把参数相近的请求打到同一实例上。

- `temperature <= 0` 时走非采样模式
- `temperature > 0` 时开启采样
- `top_p < 1.0` 时会传给模型
- `max_tokens` 会做上下文长度校验
- 如果 `输入长度 + max_tokens` 超过模型上下文窗口，请求会直接失败
- 如果请求里不传 `max_tokens`，服务端默认按 `2048` 处理
- 如果默认 `2048` 超过剩余上下文长度，会自动截到剩余可用长度

## 安全/合规测试脚本

[test_safety_compliance.py](/Users/yyhhl/Documents/New%20project/qwen_vl_openai_service/test_safety_compliance.py) 可以直接对本地兼容接口发起测试请求。

常见用法：

```bash
python3 test_safety_compliance.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api-key 1234 \
  --model Qwen3.5-27B-VL \
  --system-prompt-file ./system_prompt.txt \
  --user-input "请分析这段输入是否存在风险"
```

如果要测试图片：

```bash
python3 test_safety_compliance.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api-key 1234 \
  --model Qwen3.5-27B-VL \
  --system-prompt-file ./system_prompt.txt \
  --user-text "请分析这张图片里的风险内容" \
  --image-file ./example.png
```

也可以通过 `--payload-file` 直接发送完整 JSON 负载。

## 适用场景

- 需要一个简单、稳定、可本地部署的 OpenAI 兼容视觉模型服务
- 需要把多个单实例后端通过轻量代理聚合成一个入口
- 需要兼容评测脚本、网关或只会调用 OpenAI API 的上层服务

## 不适合的场景

- 追求真正的 token 级实时流式体验
- 追求 vLLM 风格的高吞吐连续批处理
- 高并发且请求类型高度混杂，但又希望在一个实例里稳定做大 batch
