import os
from datetime import timedelta
from pathlib import Path

import httpx
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb import MongoDBSaver
from opensandbox.config import ConnectionConfigSync
from pymongo import MongoClient

from agent.env_utils import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    ZHIPU_API_KEY, ZHIPU_BASE_URL,
)
from agent.store_compat import OffsetCompatMongoDBStore

# ---------- 模型配置 ----------
# 主 Agent 模型
MAIN_MODEL = ChatOpenAI(
    model="deepseek-v4-pro",
    temperature=1.1,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    max_tokens=2560000,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}
        }
    }
)
# 摘要专用模型（摘要需要稳定输出，temperature 设为较低值）
SUMMARY_MODEL = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.3,
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    max_tokens=2560000,
    model_kwargs={
        "extra_body": {
            "thinking": {"type": "disabled"}
        }
    }
)
# 备用模型（当主模型故障时使用）
# 注意: GLM 是智谱的模型，使用智谱的 base_url
# FALLBACK_MODEL = init_chat_model(
#     "glm-5.1",
#     model_provider="openai",
#     temperature=1.0,
#     base_url=ZHIPU_BASE_URL,
#     api_key=ZHIPU_API_KEY,
#     profile={
#         "max_input_tokens": 128000,
#         "max_output_tokens": 8192,
#         "tool_calling": True,
#         "structured_output": True,
#     }
# )

# ---------- 沙箱配置 ----------
# OpenSandbox 沙箱配置连接
# 已更新为自己的配置在012运行
SANDBOX_CONFIG = ConnectionConfigSync(
    domain="http://10.65.150.141:8080",
    use_server_proxy=True,
    request_timeout=timedelta(seconds=600),
    transport=httpx.HTTPTransport(limits=httpx.Limits(max_connections=20)),
)

# ---------- 路径常量 ----------
EXAMPLE_DIR = Path(__file__).parent.parent
print(f'当前代码执行的工作目录为：{EXAMPLE_DIR}')
# 沙箱内技能根路径
SANDBOX_SKILLS_ROOT = "/skills"
# 沙箱内分析中间文件存放目录
SANDBOX_ANALYSIS_ROOT = "/analysis"
# 沙箱内数据文件存放目录
SANDBOX_DATA_ROOT = "/data"
# 本地技能资源目录（项目内的路径，相对于项目根）
LOCAL_SKILLS_DIR = EXAMPLE_DIR / "skills"
# 本地下载目录（从沙箱下载文件的目标路径）
DOWNLOAD_DIR = EXAMPLE_DIR / "download"
# 本地子 Agent 配置目录
LOCAL_SUBAGENT_CONFIG_DIR = EXAMPLE_DIR / "agent/subagents"
# 本地的 Agent 行为准则文件（上传到沙箱 /AGENTS.md）
LOCAL_AGENTS_MD = EXAMPLE_DIR / "agent/memory/AGENTS.md"

# ---------- 文件名常量 ----------
# 主 Agent 只读指引文件（上传到沙箱 /AGENTS.md）
AGENTS_MD_FILENAME = "/AGENTS.md"

# ---------- 用户技能持久化 ----------
# 技能持久化 StoreBackend 路由路径
PERSISTED_SKILLS_ROOT = "/persisted-skills"
# 系统技能 StoreBackend 命名空间（共享，所有用户可读，由 SkillsSyncMiddleware 写入）
SYSTEM_SKILLS_STORE_NAMESPACE = ("system_skills",)
# 用户技能 StoreBackend 基命名空间（最终 namespace 在运行时追加 user_id 实现用户隔离）
USER_SKILLS_STORE_BASE_NAMESPACE = ("user_skills",)
# 子 Agent 名称 → 技能 scope 目录映射
SCOPE_MAP = {
    "main": "main",
    "procurement-analyst": "procurement",
    "procurement-order": "order",
}

# ---------- 中间件参数 ----------

# ---------- MongoDB 配置（用于持久化 Agent 短期记忆/checkpoint/长期 Store） ----------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://root:123456@127.0.0.1:27017/?authSource=admin")
MONGODB_DB_NAME = "langchain_db"
MONGODB_CHECKPOINT_COLLECTION = "checkpoints"
MONGODB_STORE_COLLECTION = "store"

_mongodb_client = MongoClient(MONGODB_URI)

# ---------- 持久化存储 ----------
# OffsetCompatMongoDBStore: persisted skills 的 MongoDB Store。
# 承载 StoreBackend 路由：/persisted-skills/system/ 与 /persisted-skills/。
# 用 offset 兼容子类，避免 StoreBackend 翻页（offset>0）触发 NotImplementedError。
STORE = OffsetCompatMongoDBStore(
    collection=_mongodb_client[MONGODB_DB_NAME][MONGODB_STORE_COLLECTION],
)

# MongoDBSaver: Agent 对话状态的 MongoDB 持久化 checkpointer。
# 支持 Human-in-the-Loop（interrupt 状态持久化）和跨重启对话恢复。
CHECKPOINTER = MongoDBSaver(
    client=_mongodb_client,
    db_name=MONGODB_DB_NAME,
    checkpoint_collection_name=MONGODB_CHECKPOINT_COLLECTION,
)

# ---------- Memory v2 MySQL 配置（方案 9 节 / 11.1 节） ----------
# DSN 无默认值、无默认密码；未启用 Memory v2 时不要求，启用后缺失即 fail closed（方案 5.8）。
MEMORY_MYSQL_DSN = os.getenv("MEMORY_MYSQL_DSN")
MEMORY_MYSQL_POOL_SIZE = int(os.getenv("MEMORY_MYSQL_POOL_SIZE", "10"))
MEMORY_MYSQL_POOL_MAX_OVERFLOW = int(os.getenv("MEMORY_MYSQL_POOL_MAX_OVERFLOW", "10"))
MEMORY_MYSQL_CONNECT_TIMEOUT = int(os.getenv("MEMORY_MYSQL_CONNECT_TIMEOUT", "5"))

# Memory v2 feature flags（方案 11.1）
# review #6：严格解析，只接受 "1"/"0"；非法字符串启动失败（fail closed），
# 避免运维误配置（"true"/"yes"/"abc"）被静默当作关闭而绕过 DSN/schema 检查


def _parse_memory_flag(name: str) -> bool:
    value = os.getenv(name, "0")
    if value not in ("0", "1"):
        raise RuntimeError(
            f"环境变量 {name} 必须是 '1' 或 '0'（当前值: {value!r}）。"
            "Memory v2 feature flag 严格解析，非法值启动失败（方案 5.8 fail closed）。"
        )
    return value == "1"


MEMORY_V2_WRITE_ENABLED = _parse_memory_flag("MEMORY_V2_WRITE_ENABLED")
MEMORY_V2_READ_ENABLED = _parse_memory_flag("MEMORY_V2_READ_ENABLED")
MEMORY_BACKGROUND_JOBS_ENABLED = _parse_memory_flag("MEMORY_BACKGROUND_JOBS_ENABLED")
MEMORY_SEMANTIC_RETRIEVAL_ENABLED = _parse_memory_flag("MEMORY_SEMANTIC_RETRIEVAL_ENABLED")

# 当前 migration head revision（健康检查校验，每次新增 migration 时同步更新，方案 5.8）
MEMORY_SCHEMA_REVISION = "0002"


