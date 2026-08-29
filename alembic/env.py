"""Alembic migration 环境（Memory v2 MySQL）。

- DSN 从 MEMORY_MYSQL_DSN 环境变量读取，未设置时 fail closed（方案 5.8）。
- 使用 async engine（asyncmy），migration 在连接上以同步方式执行。
- target_metadata 来自 src/agent/memory/orm_models.MemoryBase。
"""
from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# 导入 ORM 模型注册 metadata
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.memory.orm_models import MemoryBase  # noqa: E402

config = context.config

# DSN 由环境变量提供，禁止在配置/代码中出现默认密码（方案 9 节）
MEMORY_MYSQL_DSN = os.getenv("MEMORY_MYSQL_DSN")
if not MEMORY_MYSQL_DSN:
    raise RuntimeError(
        "MEMORY_MYSQL_DSN 未设置：Alembic migration 需要 MySQL DSN，"
        "服务 fail closed，不允许以错误配置启动（方案 5.8）。"
    )
config.set_main_option("sqlalchemy.url", MEMORY_MYSQL_DSN)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = MemoryBase.metadata


def run_migrations_offline() -> None:
    """Offline 模式：只生成 SQL，不连接数据库。"""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """异步 engine 模式：asyncmy + 每个 migration 在独立连接上执行。"""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
