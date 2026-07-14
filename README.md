# qwen_vl_openai_service

Qwen3.5 VL/OpenAI-compatible 服务部署启动说明。

## 启动服务的 Docker 命令

生产部署按机器分开启动。三台机器都先进入项目目录：

```bash
cd /root/qwen_vl_openai_service
```

### 10.2.0.130：启动 4B 后端

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host130 up -d --force-recreate --remove-orphans
```

### 10.2.0.131：启动 4B 后端

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host131 up -d --force-recreate --remove-orphans
```

### 10.2.0.129：启动 0.6B 文本后端

```bash
cd /root/qwen_vl_openai_service/qwen_text_vllm_service
docker compose --profile host129 up -d --force-recreate --remove-orphans
```

### 10.2.0.129：启动 4B 后端 + proxy

```bash
cd /root/qwen_vl_openai_service
docker compose --profile host129 up -d --force-recreate --remove-orphans
```

### 10.2.0.129：启动总入口 nginx，监听 8000

```bash
cd /root/qwen_vl_openai_service
docker compose --profile nginx up -d --force-recreate qwen35-nginx
```

## 推荐启动顺序

```text
130 -> 131 -> 129 的 0.6B -> 129 的 4B/proxy -> 129 的 nginx
```

## 对外入口

最终对外入口是：

```text
http://10.2.0.129:8000/v1
```

## 常用校验命令

### 130 校验

```bash
curl -s http://127.0.0.1:8001/ready
curl -s http://127.0.0.1:8008/ready
curl -s http://127.0.0.1:8001/health
```

### 131 校验

```bash
for p in 8001 8002 8003; do
  echo "===== 131:$p ====="
  curl -s "http://127.0.0.1:$p/ready"; echo
  curl -s "http://127.0.0.1:$p/health"; echo
done

ss -ltnp | grep -E ':800[1-3]'
ss -ltnp | grep ':8004' && echo "ERROR: 131 card-7/8004 should not exist" || echo "131:8004 removed"
```

### 129 proxy / nginx 校验

```bash
curl -s http://127.0.0.1:8005/ready
curl -s http://127.0.0.1:8008/ready
curl -s http://127.0.0.1:19000/health
curl -s http://127.0.0.1:19000/v1/models \
  -H 'Authorization: Bearer 1234'

curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models \
  -H 'Authorization: Bearer 1234'
```
