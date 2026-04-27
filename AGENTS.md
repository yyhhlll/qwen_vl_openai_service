<claude-mem-context>
# Memory Context

# [qwen_vl_openai_service] recent context, 2026-04-23 9:43pm GMT+8

Legend: 🎯session 🔴bugfix 🟣feature 🔄refactor ✅change 🔵discovery ⚖️decision
Format: ID TIME TYPE TITLE
Fetch details: get_observations([IDs]) | Search: mem-search skill

Stats: 10 obs (2,107t read) | 78,872t work | 97% savings

### Apr 23, 2026
3 10:55a 🔵 远程图片 URL 路径被 allow_remote_image_urls 标志默认封堵
4 10:56a 🔵 qwen_vl_openai_service 测试环境与项目结构
5 10:57a 🔵 ALLOW_REMOTE_IMAGE_URLS 默认关闭，协议层本地无法直接测试
6 11:05a 🟣 qwen_vl_openai_service: 默认启用远程图片 URL + 调度器按模态分批
7 11:06a 🔵 全部 6 个单元测试通过，语法编译零错误
8 11:09a ⚖️ Docker 操作规范：禁止 down，仅用非破坏性命令
9 " 🔵 qwen35_deploy.md 停止章节含 docker compose down，与新规范冲突
10 11:10a ✅ qwen35_deploy.md 停止命令替换为 docker compose stop
11 11:14a 🔵 dcu-llm-service 路径访问失败
12 " 🔵 dcu-llm-service 架构：OpenAI 兼容 VL 推理服务完整参考实现

Access 79k tokens of past work via get_observations([IDs]) or mem-search skill.
</claude-mem-context>