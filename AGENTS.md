<claude-mem-context>
# Memory Context

# [qwen_vl_openai_service] recent context, 2026-05-21 7:12pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 50 obs (6,711t read) | 457,385t work | 99% savings

### Apr 23, 2026
S2 qwen_vl_openai_service 图片过大报错的根因排查与解决方案 (Apr 23 at 10:55 PM)
32 10:57p 🔵 qwen_vl_openai_service 远程图片加载机制
S3 [**title**: Short title capturing the core action or topic] (Apr 23 at 11:01 PM)
### Apr 27, 2026
108 3:52p 🔵 提示词来源位置追踪
109 3:53p 🔵 Qwen VL 提示词处理链路追踪
110 " 🔵 Qwen VL 提示词处理完整链路
111 " 🔴 全文本批处理图像参数修复
112 4:38p 🔵 了解 Qwen VL OpenAI 服务的 nginx 反向代理配置
113 4:39p 🔵 Qwen VL API 反向代理架构：40 实例 nginx 负载均衡
### Apr 28, 2026
116 10:05a ✅ GPU服务部署模型调整
117 10:06a 🔵 docker-compose.yml 当前双实例部署结构
118 10:08a 🔵 工作会话恢复 - 继续任务
119 10:09a ✅ Docker Compose 后端缩减：24 → 12 实例
120 10:10a ✅ docker-compose.yml 完全重写：20 后端实例
121 10:11a ✅ nginx upstream 配置缩减：40 → 20 后端服务器
122 " ✅ qwen35_deploy.md 文档更新与配置验证通过
123 10:12a 🔴 单元测试失败：urlopen timeout 期望值不符
124 10:13a 🔴 测试用例修复：remote_image_timeout 期望值改为 60.0
125 " 🔴 单元测试全部通过（7/7）
126 10:14a ✅ 本次变更汇总：净减 196 行
127 10:50a 🔵 Qwen VL 服务端口状态差异：8001 正常，其他端口模型未加载
128 10:51a ✅ Engine 模型加载错误追踪机制
129 11:06a 🔵 vLLM AMD GPU部署遇到端口冲突
130 11:24a 🔵 nginx 代理配置探索
161 12:13p 🔵 探索绕过 Nginx 直接代理路由
171 12:49p ✅ Batch scheduler config refactored in server.py
172 " 🔄 Scheduler batch logic and error handling refactored
173 " 🔄 scheduler.py fully rewritten with cleaner architecture
174 12:50p 🔴 model.generate wrapped in asyncio.to_thread in engine.py
175 " 🔴 RequestState __init__ now explicitly initializes token counters
176 " 🟣 test_scheduler.py added with unit tests for Scheduler
177 12:58p 🔵 Advanced dcu-llm-service version with unified generate_batch
178 " 🔵 Advanced codebase architecture: unified generate_batch, result_future, GPU memory cleanup
179 " ✅ Docs and docker-compose updated to reflect batch inference as default
180 12:59p ✅ Session summary: qwen_vl_openai_service batch inference upgrade
181 " ✅ docker-compose.yml simplified: 1 container per GPU, removed 8009-8016 ports
182 1:10p ✅ 部署操作文档化
183 1:11p ✅ 部署文档写入完成并校验
184 1:12p ✅ BACKEND_URLS 支持环境变量覆盖
185 " ✅ 本次部署配置变更范围
186 " 🔄 docker-compose.yml 容器架构重构
187 1:45p 🔵 Qwen VL OpenAI Service 端点发现
193 6:42p 🔵 vLLM Qwen3.5-0.6B deployment on AMD GPU
194 6:44p 🟣 qwen_text_vllm_service 项目初始化
195 6:45p 🟣 qwen_text_vllm_service docker-compose 和 nginx 配置完成
196 6:46p 🟣 qwen_text_vllm_service 部署文档完成
197 6:52p ✅ 用户请求将新部分放入新文件夹
198 6:54p 🔵 项目发现新模块 qwen_text_vllm_service
199 6:55p 🔵 qwen_text_vllm_service 模块架构解析
200 " 🔄 docker-compose 后端从双实例改为单实例，削减至 19 GPU
201 " 🟣 新增 /ready 健康检查端点，支持模型加载状态检测
202 " 🔵 qwen_text_vllm_service 已自行隔离在独立目录，无需迁移

Access 457k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>