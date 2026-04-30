# Qwen3.5-0.6B 纯文本 vLLM 离线部署说明

## 1. 目标

在与原多模态项目相同的离线 DCU 服务器环境中部署纯文本 OpenAI-compatible 服务：

- 推理框架：vLLM 原生服务，不再启动本项目 Python FastAPI/Transformers VL 代码；
- 镜像：`image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250705`；
- 宿主机模型目录：`/data/model/Qwen3.5-0.6B`；
- 容器模型目录：`/models`；
- served model name：`Qwen3.5-0.6B`；
- API Key：默认 `1234`；
- 三机所有后端统一只设置 `HIP_VISIBLE_DEVICES`，不要设置 `ROCR_VISIBLE_DEVICES`；统一保留 `DEVICE_MAP=cuda`；每个后端容器还设置唯一 `MASTER_PORT`，避免 vLLM/torch 分布式初始化时在 host 网络模式下抢同一个内部通信端口。
- 单实例参考端口：`12000`；
- Compose 集群入口端口：nginx `12000`，后端固定 `12001-12004`。
- 当前资源规划：0.6B 只部署在 `10.2.0.129:g0-g3`，其余 GPU 留给 4B 多模态服务。

## 2. 部署前检查

在每台需要启动后端的机器上执行：

```bash
ls -lah /data/model/Qwen3.5-0.6B
ls -lah /opt/hyhal
```

确认镜像已在离线环境可用：

```bash
docker image inspect image.sourcefind.cn:5000/dcu/admin/base/vllm:0.8.5-ubuntu22.04-dtk25.04.1-rc5-das1.6-py3.10-20250705 >/dev/null
```

确认目标端口未被占用：

```bash
ss -lntp | grep -E ':12000|:12001|:12002|:12003|:12004' || true
```

确认三机 compose 配置没有错误的 `ROCR_VISIBLE_DEVICES`，且后端保留 `DEVICE_MAP=cuda`：

```bash
docker compose config | grep ROCR_VISIBLE_DEVICES && echo "ERROR: remove ROCR_VISIBLE_DEVICES" || echo "OK: no ROCR_VISIBLE_DEVICES"
docker compose config | grep -E 'HIP_VISIBLE_DEVICES|DEVICE_MAP'
```

## 3. 单卡验证

先用一张卡验证模型能被 vLLM 拉起：

```bash
docker compose --profile single up -d
```

查看日志：

```bash
docker logs -f qwen35-text-single-g0
```

验证 OpenAI 接口：

```bash
./scripts/smoke_chat.sh
```

或手动：

```bash
curl -s http://127.0.0.1:12000/v1/chat/completions \
  -H 'Authorization: Bearer 1234' \
  -H 'Content-Type: application/json' \
  -d @requests/chat.json
```

vLLM 正常返回时一般类似：

```json
{
  "object": "chat.completion",
  "model": "Qwen3.5-0.6B",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "reasoning_content": null,
        "content": "...",
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 303,
    "total_tokens": 311,
    "completion_tokens": 8
  }
}
```

`reasoning_content`、`tool_calls`、`stop_reason` 等字段由 vLLM 返回，属于兼容 OpenAI 接口时的扩展字段，不需要本项目额外处理。

停止单卡验证容器：

```bash
docker compose --profile single stop
```

> 注意：遵循本项目操作规范，停止用 `stop`，不要用 `docker compose down`。

## 4. 多机多 GPU 后端启动

Compose 给 nginx 入口保留 `12000`，后端实例固定使用 `10.2.0.129` 的 GPU `0-3`。

### 10.2.0.129

```bash
docker compose --profile host129 up -d
```

实例：

- `qwen35-text-129-g0` → GPU 0 → `12001`
- `qwen35-text-129-g1` → GPU 1 → `12002`
- `qwen35-text-129-g2` → GPU 2 → `12003`
- `qwen35-text-129-g3` → GPU 3 → `12004`

### 10.2.0.130 / 10.2.0.131

当前 0.6B 文本模型不再占用 `130` 或 `131` 的 GPU；这些机器的 GPU 留给 4B 多模态服务。

## 5. nginx 聚合入口

在入口机器启动 nginx：

```bash
docker compose --profile nginx up -d
```

nginx 监听：`http://<入口机器IP>:12000`

本地 nginx 进程健康检查：

```bash
curl -s http://127.0.0.1:12000/health
```

真实后端验证：

```bash
curl -s http://127.0.0.1:12000/v1/models \
  -H 'Authorization: Bearer 1234'
```

聊天验证：

```bash
BASE_URL=http://127.0.0.1:12000 API_KEY=1234 ./scripts/smoke_chat.sh
```

## 6. 常用参数

可以通过环境变量覆盖默认值：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `IMAGE_NAME` | 指定 DCU vLLM 镜像 | 如需换镜像时覆盖 |
| `MODEL_NAME` | `Qwen3.5-0.6B` | OpenAI API 返回/请求使用的模型名 |
| `API_KEY` | `1234` | Bearer token |
| `MAX_MODEL_LEN` | `4096` | vLLM `--max-model-len` |
| `GPU_MEMORY_UTILIZATION` | `0.9` | vLLM GPU 显存占用比例 |
| `EXTRA_VLLM_ARGS` | 空 | 追加 vLLM 参数 |
| `SHM_SIZE` | `200g` | Docker shm size |

示例：

```bash
MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.85 docker compose --profile host129 up -d
```

## 7. 排障

查看容器状态：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
```

查看后端日志：

```bash
docker logs --tail 200 qwen35-text-129-g0
```

检查某个后端：

```bash
curl -s http://127.0.0.1:12001/v1/models \
  -H 'Authorization: Bearer 1234'
```

常见问题：

1. **401 Unauthorized**：检查 `Authorization: Bearer 1234` 或 `API_KEY` 是否一致。
2. **端口冲突**：单卡 `single` profile 使用 `12000`，不要与 nginx 同机同时启动。
3. **模型路径错误**：确认宿主机 `/data/model/Qwen3.5-0.6B` 存在，并挂载到容器 `/models`。
4. **注意力后端报错**：本配置默认禁用 flash/triton flash，并设置 `VLLM_ATTENTION_BACKEND=TORCH_SDPA`，与用户给出的可用命令保持一致。
5. **No HIP GPUs are available**：检查 `docker compose config | grep ROCR_VISIBLE_DEVICES` 必须无输出；三机后端只保留 `HIP_VISIBLE_DEVICES`，并重建容器。
6. **EADDRINUSE / distributed 通信端口占用**：检查 `docker compose config | grep MASTER_PORT`，每个同机后端必须不同；同步新版 compose 后 `--force-recreate` 重建容器。

## 8. 停止服务

按 profile 停止，不删除资源：

```bash
docker compose --profile host129 stop
docker compose --profile host130 stop
docker compose --profile host131 stop
docker compose --profile nginx stop
```

单实例验证容器：

```bash
docker compose --profile single stop
```


## 7. 路由证明测试

从项目根目录运行：

```bash
python3 -m unittest test_proxy_routing.py
```

该测试会验证：文本合规只打 0.6B、文本不合规先 0.6B 再 4B、图片和流式请求直接 4B，以及 0.6B 只保留 `129:g0-g3` 的配置约束。
