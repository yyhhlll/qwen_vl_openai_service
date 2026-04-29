# Qwen3.5-0.6B 纯文本 vLLM OpenAI 服务

这是从 `qwen_vl_openai_service` 部署形态拆出来的纯文本版本：

- 不再加载 VL / 图片处理代码；
- 直接使用 vLLM 原生 OpenAI-compatible server；
- 适配离线 DCU 服务器；
- 默认镜像：`image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250705`；
- 默认模型目录：宿主机 `/data/model/Qwen3.5-0.6B` → 容器 `/models`；
- 默认模型名：`Qwen3.5-0.6B`；
- 默认 API Key：`1234`。
- GPU 限卡统一只使用 `HIP_VISIBLE_DEVICES`，不要设置 `ROCR_VISIBLE_DEVICES`；统一保留 `DEVICE_MAP=cuda` 作为部署约定。每个后端容器还设置唯一 `MASTER_PORT`，避免 vLLM/torch 分布式初始化时在 host 网络模式下抢同一个内部通信端口。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `docker-compose.yml` | 多机多 GPU vLLM 后端 + nginx 聚合部署 |
| `qwen35_text.nginx.conf` | nginx 负载均衡配置，监听 `12000` |
| `qwen35_text_deploy.md` | 离线服务器部署/验证/停止说明 |
| `requests/chat.json` | OpenAI `/v1/chat/completions` 示例请求 |
| `scripts/smoke_chat.sh` | 冒烟测试脚本 |

## 单实例快速启动

适合先在一张卡上验证镜像、模型、vLLM 参数是否可用：

```bash
docker run -it \
  --shm-size 200g \
  --network host \
  --name qwen35-06b \
  --privileged \
  --device=/dev/kfd \
  --device=/dev/dri \
  --device=/dev/mkfd \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -u root \
  -e DEVICE_MAP=cuda \
  -e HIP_VISIBLE_DEVICES=0 \
  -e VLLM_USE_FLASH_ATTN=0 \
  -e VLLM_USE_TRITON_FLASH_ATTN=0 \
  -e TORCH_COMPILE_DISABLE=1 \
  -e VLLM_ATTENTION_BACKEND=TORCH_SDPA \
  -v /opt/hyhal/:/opt/hyhal/:ro \
  -v /data/model/Qwen3.5-0.6B:/models:ro \
  image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250705 \
  bash -lc '
  vllm serve /models \
    --served-model-name Qwen3.5-0.6B \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 12000 \
    --max-model-len 4096 \
    --api-key 1234
  '
```

## curl 验证

```bash
curl -s http://127.0.0.1:12000/v1/chat/completions \
  -H 'Authorization: Bearer 1234' \
  -H 'Content-Type: application/json' \
  -d @requests/chat.json
```

预期返回 `object: "chat.completion"`，`model: "Qwen3.5-0.6B"`，并包含 `choices[0].message.content`。vLLM 可能额外返回 `reasoning_content`、`tool_calls`、`stop_reason` 等字段，这是正常现象。

## Compose 多机部署

> 注意：三台机器的后端实例都只设置 `HIP_VISIBLE_DEVICES`；不要再加 `ROCR_VISIBLE_DEVICES`，否则这个 DCU/vLLM 镜像可能出现 `RuntimeError: No HIP GPUs are available`。

Compose 版本给 nginx 预留 `12000`，后端实例使用 `12001` 起的端口：

- `host129`：GPU 0-7 → `12001-12008`
- `host130`：GPU 0-7 → `12001-12008`
- `host131`：GPU 4-6 → `12001-12003`
- `nginx`：监听 `12000`，转发到上述 19 个后端

启动示例：

```bash
# 10.2.0.129
docker compose --profile host129 up -d

# 10.2.0.130
docker compose --profile host130 up -d

# 10.2.0.131
docker compose --profile host131 up -d

# nginx 聚合入口，建议只在一台入口机启动
docker compose --profile nginx up -d
```

停止使用 `stop`，不要使用会删除容器/网络的破坏性命令：

```bash
docker compose --profile host129 stop
docker compose --profile nginx stop
```
