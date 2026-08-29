# Harness Agent

## 技术栈

- **Agent 框架**: LangGraph、DeepAgents
- **后端服务**: Python、FastAPI
- **工具协议**: FastMCP
- **模型服务**: DeepSeek
- **沙箱执行**: OpenSandbox
- **数据存储**: MySQL、MongoDB
- **前端**: Vue 3、Vite

## 项目结构

```text
Harness-Agent/
├── alembic/                 # 数据库迁移
├── docs/                    # 项目文档
├── frontend/                # Vue 3 前端
│   └── src/                 # 页面、组件与 API 客户端
├── src/
│   ├── agent/               # Agent 核心
│   │   ├── backends/        # OpenSandbox 后端管理
│   │   ├── memory/          # 长期记忆与任务处理
│   │   ├── middlewares/     # 召回、沙箱、工具等中间件
│   │   ├── subagents/       # 子 Agent 与 YAML 配置
│   │   ├── tools/           # MCP 工具客户端
│   │   └── main_agent.py    # 主 Agent 入口
│   ├── api_view/            # FastAPI Web API
│   ├── mcp_server/          # MCP Server 与 ERP 工具封装
│   ├── skills/              # Agent Skills
│   └── test/                # 自动化测试
├── .env.example             # 环境变量示例
├── alembic.ini              # Alembic 配置
├── langgraph.json           # LangGraph 配置
├── pyproject.toml           # Python 项目配置
├── requirements.txt         # Python 依赖
└── start_web.py             # Web 应用启动入口
```
