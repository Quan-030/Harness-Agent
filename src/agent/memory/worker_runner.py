# src/agent/memory/worker_runner.py
"""Memory v2 standalone worker 入口（方案 19.3：MEMORY_WORKER_MODE=standalone）。

用法：
    python -m agent.memory.worker_runner

与 Web 进程共享同一套启动协议（方案 5.8 fail closed）：
- 复用 MemoryDatabase.initialize：flag 合法组合 / DSN / SELECT 1 /
  alembic revision == MEMORY_SCHEMA_REVISION / hide_parameters
- 额外 gate：MEMORY_BACKGROUND_JOBS_ENABLED 必须为 1（JOBS=0 时独立 worker
  拒绝启动，避免后台自动写绕过回滚开关）；MEMORY_WORKER_MODE 必须为 standalone
- display_messages 通过 api_view.agent_loader.query_display_messages 读取
  （与 Web 进程同一提取契约，避免结构漂移）

Ctrl+C / SIGTERM 优雅停止。
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys

from pymongo import MongoClient

logger = logging.getLogger("agent.memory.worker_runner")


def validate_standalone_conditions(
    *,
    background_jobs_enabled: bool,
    worker_mode_value: str,
) -> None:
    """standalone worker 前置 gate（独立纯函数，便于测试）。

    - JOBS=0：按 feature flag 语义应停止后台记忆写入，worker 拒绝启动
    - mode != standalone：当前进程不是独立 worker 部署
    """
    if not background_jobs_enabled:
        raise RuntimeError(
            "MEMORY_BACKGROUND_JOBS_ENABLED=0：后台任务已关闭，拒绝启动 standalone worker"
        )
    if worker_mode_value != "standalone":
        raise RuntimeError(
            f"MEMORY_WORKER_MODE={worker_mode_value!r}：standalone worker 需要 "
            "MEMORY_WORKER_MODE=standalone"
        )


def _display_messages_loader(mongodb_uri: str):
    """MongoDB 直读展示消息（与 AgentLoader 同一提取契约，方案 6.1）。"""
    from api_view.agent_loader import query_display_messages

    client = MongoClient(mongodb_uri)
    collection = client["langchain_db"]["session_display_messages"]

    def _loader(thread_id: str):
        return query_display_messages(collection, thread_id) or []

    return _loader


async def _run() -> None:
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
        MONGODB_URI,
    )
    from agent.memory.database import MemoryDatabase
    from agent.memory.policies import MemoryPolicy
    from agent.memory.repository import MySQLMemoryRepository
    from agent.memory.service import MemoryService
    from agent.memory.worker import MemoryWorker, worker_mode

    validate_standalone_conditions(
        background_jobs_enabled=MEMORY_BACKGROUND_JOBS_ENABLED,
        worker_mode_value=worker_mode(),
    )

    # 与 Web 进程完全相同的初始化协议（flag 校验 / DSN / health / revision / hide_parameters）
    memory_database = MemoryDatabase()
    enabled = await memory_database.initialize(
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
    if not enabled:
        raise RuntimeError("Memory v2 未初始化，standalone worker 无法启动")

    repository = MySQLMemoryRepository(memory_database.session_factory)
    worker = MemoryWorker(
        repository,
        memory_service=MemoryService(repository, MemoryPolicy()),
        display_messages_loader=_display_messages_loader(MONGODB_URI),
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)

    logger.info("Memory standalone worker 启动（worker_id=%s）", worker._worker_id)
    try:
        await worker.run_forever(stop_event)
    finally:
        await memory_database.dispose()
        logger.info("Memory standalone worker 已退出")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
