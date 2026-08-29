# HarnessAgent — ERP 采购智能助手

基于 LangGraph + DeepAgents 的 ERP 采购智能助手，支持供应商查询、零部件管理、库存预警、订单创建等业务场景。

## 技术栈

- **框架**: LangGraph + DeepAgents + FastAPI
- **模型**: DeepSeek v4（主模型 + 摘要模型）
- **沙箱**: OpenSandbox（代码执行隔离）
- **存储**: MongoDB（checkpoint 持久化 + 沙箱注册）
- **前端**: Vue 3 + Vite
- **MCP**: FastMCP（工具协议桥接 Java 后端）

## 依赖服务

| 服务 | 端口 | 说明 |
|------|------|------|
| Java 后端 | 8080 | 业务 API，系统服务 |
| MongoDB | 27017 | 对话状态 + 沙箱注册，系统服务 |
| MCP Server | 8000 | 桥接 Java 后端，需手动启动 |
| OpenSandbox | 10.65.150.141:8080 | 代码执行沙箱，远程服务 |
| Web 后端 | 8090 | FastAPI，start_web.py 自动启动 |
| Web 前端 | 3000 | Vue dev server，start_web.py 自动启动 |

## 快速启动

### 1. 配置环境变量

```bash
cp .env .env
```

编辑 `.env`，填入必填项：

```env
# 必填 — DeepSeek API
DEEPSEEK_API_KEY=你的API_KEY
DEEPSEEK_BASE_URL=你的API地址
```

### 2. 启动 MCP Server（终端 1）

```bash
conda activate harness
cd /home/czd023/gx/ERP_OPENCLAW
python -m mcp_server.server_main
```

看到 MCP Server 在 `http://127.0.0.1:8000/mcp` 就绪即可。

### 3. 启动 Web 应用（终端 2）

```bash
conda activate harness
cd /home/czd023/gx/ERP_OPENCLAW
python start_web.py
```

等待 30-60 秒 Agent 初始化完成（首次需创建沙箱，可能更长）。

### 4. 访问

| 页面 | 地址 |
|------|------|
| 前端界面 | http://localhost:3000 |
| API 文档 | http://localhost:8090/docs |

按 `Ctrl+C` 停止所有服务。

## 项目结构

```
ERP_OPENCLAW/
├── start_web.py              # 一键启动脚本
├── .env                      # 环境变量配置
├── langgraph.json            # LangGraph 配置
├── src/
│   ├── agent/                # Agent 核心
│   │   ├── main_agent.py     # 主 Agent 入口
│   │   ├── config.py         # 全局配置
│   │   ├── env_utils.py      # 环境变量加载
│   │   ├── backends/         # 沙箱后端管理
│   │   ├── middlewares/      # Agent 中间件
│   │   ├── tools/            # MCP 工具客户端
│   │   └── memory/           # Agent 记忆提示词
│   ├── api_view/             # FastAPI Web API
│   │   └── web_main.py       # API 入口
│   ├── mcp_server/           # MCP Server（桥接 Java 后端）
│   │   └── server_main.py    # MCP Server 入口
│   └── skills/               # 沙箱技能文件
└── frontend/                 # Vue 前端
```

## Memory v2 运维（MySQL 长期记忆）

Memory v2 使用 MySQL（`memory_v2` 库）作为唯一长期记忆事实源，旧 Markdown 记忆路径已删除（方案 `docs/memory-upgrade-plan.md`）。

### 首次启用（开关顺序，方案 21.1）

```bash
# 1. 备份目标 MySQL，执行 schema migration（从空库可升级）
alembic -c alembic.ini upgrade head

# 2. 验证 DSN 与 migration revision（MEMORY_SCHEMA_REVISION=0002）
# 3. 按顺序开启（.env）：
#    MEMORY_V2_WRITE_ENABLED=1      # 仅显式同步写入的预热状态
#    MEMORY_V2_READ_ENABLED=1       # 参与回答（召回中间件）
#    MEMORY_WORKER_MODE=embedded    # 先配置 worker
#    MEMORY_BACKGROUND_JOBS_ENABLED=1  # 自动抽取后台任务
#    MEMORY_SEMANTIC_RETRIEVAL_ENABLED=1  # 按需
```

非法组合（启动失败，fail closed）：`WRITE=0 且 JOBS=1`；`SEMANTIC=1 且 READ=0`。MySQL 故障时只降级（无长期记忆），**不恢复旧 Markdown**。

### Worker 启动方式

| 模式 | 说明 | 启动 |
|------|------|------|
| `embedded` | Web 进程内运行（lifespan 自动启动） | `MEMORY_WORKER_MODE=embedded` |
| `standalone` | 独立进程（推荐生产） | `MEMORY_WORKER_MODE=standalone && python -m agent.memory.worker_runner` |
| `disabled` | 默认；JOBS=1 前必须改为前两者之一，否则 job 无人消费 | — |

### 健康检查与故障排查

- 健康检查：`GET /health`（含 Memory v2 MySQL 状态）；启动时 DSN 缺失/版本不匹配即 fail closed
- dead job：`memory_jobs.status='dead'`（超过 5 次尝试），`last_error` 只含受控 reason code（方案 20.3 不保存正文）；人工处理后删除或重放
- 备份恢复：备份 MySQL `memory_v2` 库（`mysqldump`）；恢复后 revision/generation 随库一致，无需额外迁移
- 观测：运行指标事件（`memory_recall_degraded` / `memory_job_retried` 等，含 event/reason_code/outcome 字段）从应用日志聚合
