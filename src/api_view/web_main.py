"""
DeepAgent Chat API - FastAPI 主应用

提供基于 DeepAgent 的 AI 对话系统后端 API
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_view.web_config import API_TITLE, API_VERSION, API_DESCRIPTION
from api_view.api import chat, history, memory
from api_view.agent_loader import agent_loader
from agent.config import (
    MEMORY_BACKGROUND_JOBS_ENABLED,
    MEMORY_MYSQL_CONNECT_TIMEOUT,
    MEMORY_MYSQL_DSN,
    MEMORY_MYSQL_POOL_MAX_OVERFLOW,
    MEMORY_MYSQL_POOL_SIZE,
    MEMORY_SCHEMA_REVISION,
    MEMORY_SEMANTIC_RETRIEVAL_ENABLED,
    MEMORY_V2_READ_ENABLED,
    MEMORY_V2_WRITE_ENABLED,
)
from agent.memory.database import memory_database
from agent.memory.worker import worker_mode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    在应用启动时初始化 Agent，在应用关闭时清理资源
    """
    # ============================================================
    # 应用启动时执行
    # ============================================================
    print("=" * 50)
    print("正在启动 DeepAgent Chat API...")
    print("=" * 50)

    # Memory v2 MySQL 初始化（方案 5.8：启用即 fail closed；完全关闭则跳过）
    memory_enabled = await memory_database.initialize(
        dsn=MEMORY_MYSQL_DSN,
        pool_size=MEMORY_MYSQL_POOL_SIZE,
        pool_max_overflow=MEMORY_MYSQL_POOL_MAX_OVERFLOW,
        connect_timeout=MEMORY_MYSQL_CONNECT_TIMEOUT,
        expected_revision=MEMORY_SCHEMA_REVISION,
        write_enabled=MEMORY_V2_WRITE_ENABLED,
        read_enabled=MEMORY_V2_READ_ENABLED,
        jobs_enabled=MEMORY_BACKGROUND_JOBS_ENABLED,
        semantic_enabled=MEMORY_SEMANTIC_RETRIEVAL_ENABLED,
    )
    if memory_enabled:
        print("[MemoryDatabase] Memory v2 MySQL 已就绪（fail closed 校验通过）")

    # 初始化 Agent（display_messages loader 依赖 MongoDB 就绪，先于 worker 启动）
    await agent_loader.initialize()

    # Memory v2 embedded worker（方案 19.3 + review #2：MEMORY_WORKER_MODE=embedded 时
    # 在 Web 进程内启动；生产默认 standalone 独立进程）。
    # 注入真实依赖：MemoryService（Policy/命令准备）+ display_messages_loader
    worker_task = None
    if memory_enabled and MEMORY_BACKGROUND_JOBS_ENABLED and worker_mode() == "embedded":
        from agent.memory.policies import MemoryPolicy
        from agent.memory.repository import MySQLMemoryRepository
        from agent.memory.service import MemoryService
        from agent.memory.worker import MemoryWorker

        memory_repo = MySQLMemoryRepository(memory_database.session_factory)
        memory_worker = MemoryWorker(
            memory_repo,
            memory_service=MemoryService(memory_repo, MemoryPolicy()),
            display_messages_loader=agent_loader.get_display_messages,
        )
        worker_stop = asyncio.Event()
        worker_task = asyncio.create_task(memory_worker.run_forever(worker_stop))
        print("[MemoryWorker] embedded worker 已启动")

    print("=" * 50)
    print("DeepAgent Chat API 启动成功!")
    print("=" * 50)

    # 继续运行
    yield

    # ============================================================
    # 应用关闭时执行
    # ============================================================
    print("=" * 50)
    print("正在关闭 DeepAgent Chat API...")
    print("=" * 50)

    # 停止 embedded worker（给当前任务最多 30 秒完成，方案 19.3）
    if worker_task is not None:
        worker_stop.set()
        try:
            await asyncio.wait_for(worker_task, timeout=30)
        except asyncio.TimeoutError:
            worker_task.cancel()
            print("[MemoryWorker] embedded worker 停止超时，已取消任务")

    # 清理所有沙箱 + MongoDB 连接
    await agent_loader.shutdown()

    # 关闭 Memory v2 MySQL 连接池
    await memory_database.dispose()

    print("DeepAgent Chat API 已关闭")


# 创建 FastAPI 应用
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

# ============================================================
# Memory API 异常映射（review K / 3.4 / 3.5：生产与测试共用）
# ============================================================

from fastapi.exception_handlers import (  # noqa: E402
    request_validation_exception_handler,
)
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from sqlalchemy.exc import OperationalError  # noqa: E402

# Memory API 路径前缀（review 3.4：handler 只改写 Memory 路由的校验错误）
_MEMORY_API_PREFIXES = ("/api/memory/", "/api/memories")


def register_memory_exception_handlers(app: FastAPI) -> None:
    """注册 Memory API 的异常映射（review 3.5：生产 app 与测试 app 共用同一逻辑）。

    - Memory API 请求校验失败 → 400（方案 20.2）
    - 其他路由（chat/history）保持 FastAPI 默认 422（review 3.4）
    - MySQL 运行中不可用 → 503
    """

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request, exc: RequestValidationError):
        path = request.url.path
        if path.startswith(_MEMORY_API_PREFIXES):
            return JSONResponse(status_code=400, content={"detail": "请求参数校验失败"})
        return await request_validation_exception_handler(request, exc)

    @app.exception_handler(OperationalError)
    async def operational_error_handler(request, exc: OperationalError):
        return JSONResponse(status_code=503, content={"detail": "MySQL 暂不可用，请稍后重试"})


register_memory_exception_handlers(app)

# ============================================================
# CORS 中间件配置
# ============================================================
# 允许跨域请求，方便前端开发
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 注册路由
# ============================================================
# 对话相关接口
app.include_router(chat.router, prefix="/api", tags=["对话"])
# 历史记录相关接口
app.include_router(history.router, prefix="/api", tags=["历史记录"])
# Memory v2 记忆接口
app.include_router(memory.router, prefix="/api", tags=["记忆"])


# ============================================================
# 根路径
# ============================================================
@app.get("/", tags=["首页"])
async def root():
    """
    根路径

    返回 API 基本信息
    """
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": "基于 DeepAgent 的 AI 对话系统 API",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# ============================================================
# 健康检查
# ============================================================
@app.get("/health", tags=["系统"])
async def health_check():
    """
    健康检查接口

    用于检查服务是否正常运行
    """
    return {
        "status": "healthy",
        "service": API_TITLE,
        "version": API_VERSION
    }


# ============================================================
# 启动命令
# ============================================================
# uvicorn api_view.web_main:app --reload --host 0.0.0.0 --port 8000
