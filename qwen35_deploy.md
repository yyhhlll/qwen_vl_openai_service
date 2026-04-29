# Qwen3.5-4B 三机部署

当前默认入口：`qwen35-proxy`，监听 `10.2.0.129:19000`。

> 现在生产调用优先直接走 proxy，不走 nginx。`qwen35.nginx.conf` 仍保留为可选入口配置，但不是默认部署路径。

## 分配

- `129`
  - GPU `0-7`
  - 每张卡 1 个服务
  - 后端端口 `8001-8008`
  - `proxy` 入口 `19000`
- `130`
  - GPU `0-7`
  - 每张卡 1 个服务
  - 后端端口 `8001-8008`
- `131`
  - GPU `4-6`
  - 每张卡 1 个服务
  - 后端端口 `8001-8003`
  - 不部署 7 号卡；`8004` 不再属于 131

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

`129` 只保留 `qwen35-129-g0..g7` 和 `qwen35-proxy`：

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

- `qwen35-129-g0` ... `qwen35-129-g7`
- `qwen35-proxy`

如果只更新了 proxy 配置或只想重启 proxy：

```bash
docker compose --profile proxy up -d --force-recreate qwen35-proxy
```

校验本机后端和 proxy：

```bash
curl -s http://127.0.0.1:8001/ready
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
docker logs --tail 100 qwen35-129-g0
curl -s http://127.0.0.1:8001/ready
curl -s http://127.0.0.1:19000/health
```

## 验证

后端：

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8001/ready
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

proxy 会按 `BACKEND_URLS` 在所有后端之间转发。若某些后端未 ready，聊天请求可能被分配到未就绪端口并超时。

先看 proxy 汇总健康：

```bash
curl -s http://127.0.0.1:19000/health
```

再逐台逐端口查 ready：

```bash
# 129
for p in 8001 8002 8003 8004 8005 8006 8007 8008; do
  echo "129:$p"; curl -s "http://127.0.0.1:$p/ready"; echo
done

# 130 在对应机器本机检查 8001-8008；131 只检查 8001-8003
```

如果只想临时让 proxy 转发到 129 本机后端，可在启动 proxy 时覆盖后端池：

```bash
BACKEND_URLS=http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003,http://127.0.0.1:8004,http://127.0.0.1:8005,http://127.0.0.1:8006,http://127.0.0.1:8007,http://127.0.0.1:8008 \
docker compose --profile proxy up -d --force-recreate qwen35-proxy
```

### 可选：nginx 入口

当前默认不走 nginx。如需启用 nginx，可单独启动：

```bash
docker compose --profile nginx up -d --force-recreate qwen35-nginx
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/nginx_status
```

如果 nginx 日志里仍出现 `8009-8016` 或 `10.2.0.131:8004-8008`，说明 nginx 容器还在使用旧 upstream。先确认宿主机配置文件已经是新版：

```bash
grep -nE '8009|8010|8011|8012|8013|8014|8015|8016|10\.2\.0\.131:800[4-8]' qwen35.nginx.conf || echo "host nginx config clean"
```

再强制重建 nginx 容器：

```bash
docker compose --profile nginx up -d --force-recreate qwen35-nginx
```

最后确认容器内实际加载的配置也没有旧端口：

```bash
docker exec qwen35-nginx nginx -T 2>/dev/null | \
  grep -E '8009|8010|8011|8012|8013|8014|8015|8016|10\.2\.0\.131:800[4-8]' \
  || echo "nginx upstream clean"
```
