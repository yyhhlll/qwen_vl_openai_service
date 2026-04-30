# Qwen3.5-4B 三机部署

当前推荐对外入口：`qwen35-nginx`，监听 `10.2.0.129:8000`，并转发到 `qwen35-proxy` 的 `19000`。

> `qwen35.nginx.conf` 现在只作为总入口转发到 `qwen35-proxy:19000`；0.6B/4B 两阶段路由由 proxy.py 决定。不要让总入口 nginx 直接转发到 4B 后端池，否则文本不会先走 0.6B。

## 分配

当前 19 张卡拆分为：0.6B 纯文本模型使用 `129:g0-g3`，4B 多模态模型使用剩余 15 张卡。

- `129`
  - 4B 使用 GPU `4-7`
  - 每张卡 1 个 4B 服务
  - 4B 后端端口 `8005-8008`
  - 0.6B 文本服务使用 GPU `0-3`，端口 `12001-12004`（见 `qwen_text_vllm_service/qwen35_text_deploy.md`）
  - `proxy` 内部入口 `19000`
  - `nginx` 对外入口 `8000`
- `130`
  - 4B 使用 GPU `0-7`
  - 每张卡 1 个 4B 服务
  - 4B 后端端口 `8001-8008`
- `131`
  - 4B 使用 GPU `4-6`
  - 每张卡 1 个 4B 服务
  - 4B 后端端口 `8001-8003`
  - 不部署 7 号卡；`8004` 不再属于 131

## proxy 两阶段路由

`qwen35-proxy` 仍对外暴露一个 OpenAI-compatible 入口，但内部维护两个后端池：

- `TEXT_BACKEND_URLS`：0.6B 文本池，默认 `http://10.2.0.129:12001-12004`
- `BACKEND_URLS`：4B 多模态池，默认 `129:8005-8008 + 130:8001-8008 + 131:8001-8003`

路由规则：

- 文本非流式请求先调用 0.6B；0.6B 明确返回合规时直接返回该结果。
- 发给 0.6B 的首段请求体会被收敛为兼容最小格式：`model`、`messages`、`max_tokens`；不会把 `response_format`、`top_p`、`stop`、`seed`、`presence_penalty`、显式 `stream=false` 等 4B/客户端参数传给 0.6B。纯文本 content part 列表会压平成字符串。
- 文本非流式请求若 0.6B 返回不合规、异常、模糊、非 2xx 或无法解析，则用原始请求体升级调用 4B 返回类别结果，只改写 4B 内部 `model`。
- 图片/多模态请求直接调用 4B。
- 所有 `stream=true` 请求直接调用 4B，保持 SSE 字节流透传；流式 chunk 中的 `model` 字段不做归一化保证。
- `/v1/models` 返回确定性的 0.6B + 4B 模型列表，不再随机代理某一个后端。


### 路由证明测试

部署前后可在代码目录运行离线单元测试，证明三类路由和模型名重写逻辑：

```bash
python3 -m unittest test_proxy_routing.py
```

完整回归：

```bash
python3 -m unittest test_proxy_routing.py test_image_url_support.py test_engine_memory_cleanup.py
```

## 目录

三台机器都使用：

```bash
cd /root/qwen_vl_openai_service
```

离线默认已开启：

- `OFFLINE_MODE=1`
- `ALLOW_REMOTE_IMAGE_URLS=1`
- `REMOTE_IMAGE_TIMEOUT_SECONDS=60`
- `LOAD_MODEL_ON_STARTUP=1`
- `DEVICE_MAP=cuda`
- `MAX_BATCH_SIZE=4`
- `BATCH_WAIT_MS=50`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## 启动

建议启动顺序：先启动纯后端机器 `130` / `131`，确认 ready 后，再启动 `129` 后端和 proxy。proxy 的后端池包含三台机器，如果其他机器还没 ready，业务请求可能被转发到未就绪后端而超时。

### 三台机器同步新版后的一次性重建命令

三台机器都先进入项目目录：

```bash
cd /root/qwen_vl_openai_service
```

`129` 的 4B 只保留 `qwen35-129-g4..g7` 和 `qwen35-proxy`；`g0..g3` 留给 0.6B 文本服务：

```bash
docker stop \
  qwen35-4b \
  qwen35-129-g0a qwen35-129-g0b qwen35-129-g1a qwen35-129-g1b \
  qwen35-129-g2a qwen35-129-g2b qwen35-129-g3a qwen35-129-g3b \
  qwen35-129-g4a qwen35-129-g4b qwen35-129-g5a qwen35-129-g5b \
  qwen35-129-g6a qwen35-129-g6b qwen35-129-g7a qwen35-129-g7b \
  qwen35-130-g0 qwen35-130-g1 qwen35-130-g2 qwen35-130-g3 \
  qwen35-130-g4 qwen35-130-g5 qwen35-130-g6 qwen35-130-g7 \
  qwen35-130-g0b qwen35-130-g1b qwen35-130-g2b qwen35-130-g3b \
  qwen35-130-g4b qwen35-130-g5b qwen35-130-g6b qwen35-130-g7b \
  qwen35-131-g4 qwen35-131-g5 qwen35-131-g6 qwen35-131-g7 \
  qwen35-131-g4b qwen35-131-g5b qwen35-131-g6b qwen35-131-g7b \
  2>/dev/null || true

docker compose --profile host129 up -d --force-recreate --remove-orphans
```

`130` 只保留 `qwen35-130-g0..g7`：

```bash
docker stop \
  qwen35-proxy qwen35-4b \
  qwen35-129-g0 qwen35-129-g1 qwen35-129-g2 qwen35-129-g3 \
  qwen35-129-g4 qwen35-129-g5 qwen35-129-g6 qwen35-129-g7 \
  qwen35-129-g0a qwen35-129-g0b qwen35-129-g1a qwen35-129-g1b \
  qwen35-129-g2a qwen35-129-g2b qwen35-129-g3a qwen35-129-g3b \
  qwen35-129-g4a qwen35-129-g4b qwen35-129-g5a qwen35-129-g5b \
  qwen35-129-g6a qwen35-129-g6b qwen35-129-g7a qwen35-129-g7b \
  qwen35-130-g0b qwen35-130-g1b qwen35-130-g2b qwen35-130-g3b \
  qwen35-130-g4b qwen35-130-g5b qwen35-130-g6b qwen35-130-g7b \
  qwen35-131-g4 qwen35-131-g5 qwen35-131-g6 qwen35-131-g7 \
  qwen35-131-g4b qwen35-131-g5b qwen35-131-g6b qwen35-131-g7b \
  2>/dev/null || true

docker compose --profile host130 up -d --force-recreate --remove-orphans
```

`131` 只保留 `qwen35-131-g4..g6`：

```bash
docker stop \
  qwen35-proxy qwen35-4b \
  qwen35-129-g0 qwen35-129-g1 qwen35-129-g2 qwen35-129-g3 \
  qwen35-129-g4 qwen35-129-g5 qwen35-129-g6 qwen35-129-g7 \
  qwen35-129-g0a qwen35-129-g0b qwen35-129-g1a qwen35-129-g1b \
  qwen35-129-g2a qwen35-129-g2b qwen35-129-g3a qwen35-129-g3b \
  qwen35-129-g4a qwen35-129-g4b qwen35-129-g5a qwen35-129-g5b \
  qwen35-129-g6a qwen35-129-g6b qwen35-129-g7a qwen35-129-g7b \
  qwen35-130-g0 qwen35-130-g1 qwen35-130-g2 qwen35-130-g3 \
  qwen35-130-g4 qwen35-130-g5 qwen35-130-g6 qwen35-130-g7 \
  qwen35-130-g0b qwen35-130-g1b qwen35-130-g2b qwen35-130-g3b \
  qwen35-130-g4b qwen35-130-g5b qwen35-130-g6b qwen35-130-g7b \
  qwen35-131-g7 qwen35-131-g4b qwen35-131-g5b qwen35-131-g6b qwen35-131-g7b \
  2>/dev/null || true

docker rm -f qwen35-131-g7 qwen35-131-g7b 2>/dev/null || true

docker compose --profile host131 up -d --force-recreate --remove-orphans
```

各机器重建后确认没有错误 profile 的旧容器：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Command}}' | grep qwen35
```

### 130：启动后端

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host130 up -d --force-recreate
```

校验：

```bash
curl -s http://127.0.0.1:8001/ready
curl -s http://127.0.0.1:8008/ready
curl -s http://127.0.0.1:8001/health
```

### 131：启动后端

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host131 up -d --force-recreate --remove-orphans
```

校验 131 仅检查 `8001-8003`，不要再检查 `8004`：

```bash
for p in 8001 8002 8003; do
  echo "===== 131:$p ====="
  curl -s "http://127.0.0.1:$p/ready"; echo
  curl -s "http://127.0.0.1:$p/health"; echo
done

ss -ltnp | grep -E ':800[1-3]'
ss -ltnp | grep ':8004' && echo "ERROR: 131 card-7/8004 should not exist" || echo "131:8004 removed"
```

### 129：启动后端 + proxy

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host129 up -d --force-recreate
```

`host129` profile 会同时启动：

- `qwen35-129-g4` ... `qwen35-129-g7`
- `qwen35-proxy`

如果只更新了 proxy 配置或只想重启 proxy：

```bash
docker compose --profile proxy up -d --force-recreate qwen35-proxy
```

校验本机后端和 proxy：

```bash
curl -s http://127.0.0.1:8005/ready
curl -s http://127.0.0.1:8008/ready
curl -s http://127.0.0.1:19000/health
curl -s http://127.0.0.1:19000/v1/models \
  -H 'Authorization: Bearer 1234'
```

## 停止

```bash
docker compose --profile host129 stop
docker compose --profile host130 stop
docker compose --profile host131 stop
```

如果机器上残留旧版每卡第二个服务（旧容器名带 `b`，或旧 `129` 容器名带 `a`），先用非破坏性的 `docker stop` 停掉旧容器，再启动新版配置：

```bash
docker stop \
  qwen35-129-g0a qwen35-129-g0b qwen35-129-g1a qwen35-129-g1b \
  qwen35-129-g2a qwen35-129-g2b qwen35-129-g3a qwen35-129-g3b \
  qwen35-129-g4a qwen35-129-g4b qwen35-129-g5a qwen35-129-g5b \
  qwen35-129-g6a qwen35-129-g6b qwen35-129-g7a qwen35-129-g7b \
  qwen35-130-g0b qwen35-130-g1b qwen35-130-g2b qwen35-130-g3b \
  qwen35-130-g4b qwen35-130-g5b qwen35-130-g6b qwen35-130-g7b \
  qwen35-131-g7 qwen35-131-g4b qwen35-131-g5b qwen35-131-g6b qwen35-131-g7b \
  2>/dev/null || true
```

## 检查

```bash
docker compose ps
docker logs --tail 100 qwen35-proxy
docker logs --tail 100 qwen35-129-g4
curl -s http://127.0.0.1:8005/ready
curl -s http://127.0.0.1:19000/health
```

## 验证

后端：

```bash
curl -s http://127.0.0.1:8005/health
curl -s http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8005/ready
curl -s http://127.0.0.1:8008/ready
```

重点看：

- `binding.hip_visible_devices`
- `binding.model_device`
- `binding.device_map`
- `loaded`
- `last_load_error`

说明：

- `/health` 只代表进程存活，并会展示模型加载状态。
- `/ready` 才代表模型已经加载完成；只有返回 `200` 且 `loaded=true` 的端口才应该接业务流量。
- `/v1/models` 是兼容接口的模型列表，不触发模型加载，能返回不代表该端口已经 ready。
- 单个后端会在 `BATCH_WAIT_MS` 短窗口内最多收集 `MAX_BATCH_SIZE` 个同类请求，一次调用模型做 batch 推理；纯文本和图文请求会拆成不同子批次执行。
- `binding.model_device` 必须是 `cuda:*`；如果是 `cpu`，聊天请求会非常慢，看起来像卡死。默认 `DEVICE_MAP=cuda` 会强制加载到当前容器可见的 DCU/GPU。
- 当前镜像限卡只使用 `HIP_VISIBLE_DEVICES`。不要同时设置 `ROCR_VISIBLE_DEVICES`：实测 `HIP_VISIBLE_DEVICES=7` 可让 torch 看到 1 张卡，`ROCR_VISIBLE_DEVICES=7` 会让 torch 仍看到 8 张卡，二者组合会导致 HIP 初始化失败。

proxy：

```bash
curl -s http://127.0.0.1:19000/health

curl -s http://127.0.0.1:19000/v1/models \
  -H 'Authorization: Bearer 1234'
```

聊天测试：

```bash
curl -s http://127.0.0.1:19000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer 1234' \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [
      {"role": "user", "content": "你好，回复一个ok"}
    ],
    "max_tokens": 16
  }'
```

外部客户端入口：

```bash
http://10.2.0.129:19000/v1/chat/completions
```

## 排障

### `/ready` 返回 Not Found

这通常说明该端口还跑着旧版代码。新版服务的 `/health` 会包含：

- `loaded`
- `loading`
- `last_load_error`
- `load_model_on_startup`

如果 `/ready` 是 `{"detail":"Not Found"}`，同时 `/health` 没有这些字段，说明请求打到了旧进程。

### 新容器日志显示 `address already in use`

这说明旧容器或旧进程还占着端口，新容器没有真正启动成功。例如新版 `qwen35-129-g7` 使用 `8008`，旧版 `qwen35-129-g3b` 也使用 `8008`。

先定位端口占用：

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Command}}' | grep qwen35
ss -ltnp | grep ':8008'
```

然后停掉旧版容器，再启动新版配置：

```bash
docker stop qwen35-129-g3b qwen35-129-g7b 2>/dev/null || true
docker compose --profile host129 up -d
docker logs --tail 100 qwen35-129-g7
curl -s http://127.0.0.1:8008/ready
```

### proxy 请求超时

proxy 会按 `BACKEND_URLS` 在 4B 后端池之间转发，并按 `TEXT_BACKEND_URLS` 调用 0.6B 文本池。若某些后端未 ready，聊天请求可能被分配到未就绪端口并超时。

先看 proxy 汇总健康：

```bash
curl -s http://127.0.0.1:19000/health
```

再逐台逐端口查 ready：

```bash
# 129 上 4B 只检查 8005-8008；8001-8004 留给已移除的旧 4B/0.6B 规划，不应作为 4B 后端
for p in 8005 8006 8007 8008; do
  echo "129:$p"; curl -s "http://127.0.0.1:$p/ready"; echo
done

# 129 上 0.6B 文本池检查 12001-12004
for p in 12001 12002 12003 12004; do
  echo "129-text:$p"; curl -s "http://127.0.0.1:$p/v1/models" -H 'Authorization: Bearer 1234'; echo
done

# 130 在对应机器本机检查 8001-8008；131 只检查 8001-8003
```

如果只想临时让 proxy 转发到 129 本机后端，可在启动 proxy 时覆盖后端池。注意 4B 只能使用 `8005-8008`，0.6B 文本池使用 `12001-12004`：

```bash
BACKEND_URLS=http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007,http://127.0.0.1:8008 \
TEXT_BACKEND_URLS=http://127.0.0.1:12001,http://127.0.0.1:12002,http://127.0.0.1:12003,http://127.0.0.1:12004 \
docker compose --profile proxy up -d --force-recreate qwen35-proxy
```

### nginx 总入口

总入口 nginx 必须转发到 `qwen35-proxy:19000`。启动：

```bash
docker compose --profile nginx up -d --force-recreate qwen35-nginx
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models -H 'Authorization: Bearer 1234'
curl -s http://127.0.0.1:8000/nginx_status
```

如果 nginx 配置里仍出现 `qwen35_backend`、`10.2.0.*:800*` 直连 4B upstream，说明总入口仍会绕过 proxy。先确认宿主机配置文件已经是新版：

```bash
grep -nE 'qwen35_backend|10\.2\.0\.(129|130|131):800[0-9]' qwen35.nginx.conf && echo "ERROR: nginx bypasses proxy" || echo "host nginx routes only to proxy"
```

再强制重建 nginx 容器：

```bash
docker compose --profile nginx up -d --force-recreate qwen35-nginx
```

最后确认容器内实际加载的配置也只转发到 proxy：

```bash
docker exec qwen35-nginx nginx -T 2>/dev/null | \
  grep -nE 'qwen35_proxy_gateway|127\.0\.0\.1:19000|proxy_pass http://qwen35_proxy_gateway'
```
