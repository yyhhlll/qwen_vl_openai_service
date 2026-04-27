# Qwen3.5-4B 三机部署

## 分配

- `129`
  - GPU `0-7`
  - 每张卡 2 个服务
  - 后端端口 `8001-8016`
  - `nginx` 入口 `8000`
- `130`
  - GPU `0-7`
  - 每张卡 2 个服务
  - 后端端口 `8001-8016`
- `131`
  - GPU `4-7`
  - 每张卡 2 个服务
  - 后端端口 `8001-8008`

## 目录

三台机器都使用：

```bash
cd /root/qwen_vl_openai_service
```

离线默认已开启：

- `OFFLINE_MODE=1`
- `ALLOW_REMOTE_IMAGE_URLS=1`
- `REMOTE_IMAGE_TIMEOUT_SECONDS=60`
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`

## 启动

### 129

```bash
docker compose --profile host129 up -d
docker compose --profile nginx up -d qwen35-nginx
```

如果离线环境没有 `nginx:stable-alpine`，改用本地已有镜像：

```bash
NGINX_IMAGE=<local-nginx-image> docker compose --profile nginx up -d qwen35-nginx
```

如果只更新了 [qwen35.nginx.conf](/Users/yyhhl/Documents/New project/qwen_vl_openai_service/qwen35.nginx.conf)，可直接重启 nginx 容器生效：

```bash
docker compose restart qwen35-nginx
```

### 130

```bash
docker compose --profile host130 up -d
```

### 131

```bash
docker compose --profile host131 up -d
```

## 停止

```bash
docker compose --profile host129 stop
docker compose --profile host130 stop
docker compose --profile host131 stop
```

## 检查

```bash
docker compose ps
docker logs --tail 100 qwen35-nginx
docker logs --tail 100 qwen35-129-g0a
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/nginx_status
```

## 验证

后端：

```bash
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8016/health
curl -s http://127.0.0.1:8008/health
```

重点看：

- `binding.hip_visible_devices`
- `binding.rocr_visible_devices`
- `binding.model_device`

nginx：

```bash
curl -s http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer 1234'
```

聊天测试：

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
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
