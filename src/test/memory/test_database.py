# src/test/memory/test_database.py
"""Memory v2 数据库生命周期与 fail closed 语义测试（方案 5.8 / 21.2）。

- feature flag 非法组合拒绝（21.2）
- 完全关闭时跳过初始化，不要求 DSN
- 启用时 DSN 缺失 / 健康检查失败 → fail closed
- migration 版本不匹配 → fail closed
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.database import (  # noqa: E402
    MemoryDatabase,
    check_memory_database_health,
    memory_v2_enabled,
    validate_memory_flags,
)

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_health_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

FLAGS = dict(
    write_enabled=False,
    read_enabled=False,
    jobs_enabled=False,
    semantic_enabled=False,
)


# ============================================================
# flag 合法性（方案 21.2）
# ============================================================

def test_all_off_is_legal():
    validate_memory_flags(**FLAGS)
    assert memory_v2_enabled(**FLAGS) is False


def test_full_v2_is_legal():
    flags = dict(FLAGS, write_enabled=True, read_enabled=True, jobs_enabled=True)
    validate_memory_flags(**flags)
    assert memory_v2_enabled(**flags) is True


def test_jobs_without_write_illegal():
    with pytest.raises(RuntimeError):
        validate_memory_flags(**dict(FLAGS, jobs_enabled=True))


def test_jobs_without_read_illegal():
    """WRITE=1 READ=0 JOBS=1：自动记忆不可见，非法。"""
    with pytest.raises(RuntimeError):
        validate_memory_flags(
            **dict(FLAGS, write_enabled=True, jobs_enabled=True)
        )


def test_semantic_without_read_illegal():
    with pytest.raises(RuntimeError):
        validate_memory_flags(**dict(FLAGS, semantic_enabled=True))


def test_write_only_prewarm_is_legal():
    """WRITE=1 READ=0 JOBS=0：仅显式同步写入的预热状态（21.2 合法）。"""
    validate_memory_flags(**dict(FLAGS, write_enabled=True))


def test_flag_strict_parsing(monkeypatch):
    """review #6：flag 只接受 '1'/'0'，非法字符串启动失败（fail closed）。"""
    from agent.config import _parse_memory_flag

    monkeypatch.setenv("MEMORY_TEST_FLAG", "1")
    assert _parse_memory_flag("MEMORY_TEST_FLAG") is True
    monkeypatch.setenv("MEMORY_TEST_FLAG", "0")
    assert _parse_memory_flag("MEMORY_TEST_FLAG") is False
    for bad in ("true", "TRUE", "yes", "abc", "on"):
        monkeypatch.setenv("MEMORY_TEST_FLAG", bad)
        with pytest.raises(RuntimeError):
            _parse_memory_flag("MEMORY_TEST_FLAG")


# ============================================================
# initialize 生命周期
# ============================================================

def test_initialize_skips_when_fully_disabled():
    """完全关闭：跳过初始化，不要求 DSN，返回 False。"""
    db = MemoryDatabase()

    async def _run():
        return await db.initialize(
            dsn=None,  # 无 DSN 也应正常跳过
            pool_size=5,
            pool_max_overflow=5,
            connect_timeout=5,
            expected_revision="0002",
            **FLAGS,
        )

    assert asyncio.run(_run()) is False
    assert db.initialized is False


def test_initialize_fails_closed_without_dsn():
    """启用但 DSN 缺失 → fail closed。"""
    db = MemoryDatabase()

    async def _run():
        return await db.initialize(
            dsn=None,
            pool_size=5,
            pool_max_overflow=5,
            connect_timeout=5,
            expected_revision="0002",
            **dict(FLAGS, read_enabled=True),
        )

    with pytest.raises(RuntimeError, match="MEMORY_MYSQL_DSN"):
        asyncio.run(_run())
    assert db.initialized is False


# ============================================================
# 健康检查（migration 版本匹配，方案 5.8）
# ============================================================

@pytest.fixture()
def migrated_health_db():
    """独立测试库：upgrade 到 head 后执行测试，teardown 删除。"""
    async def _create():
        engine = create_async_engine(BASE_URL)
        async with engine.begin() as conn:
            await conn.execute(
                text(f"CREATE DATABASE IF NOT EXISTS {TEST_DB} CHARACTER SET utf8mb4")
            )
        await engine.dispose()

    async def _drop():
        engine = create_async_engine(BASE_URL)
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
        await engine.dispose()

    asyncio.run(_create())
    cfg = Config(ALEMBIC_INI)
    os.environ["MEMORY_MYSQL_DSN"] = DSN
    command.upgrade(cfg, "head")
    yield
    asyncio.run(_drop())


def test_health_check_passes_with_matching_revision(migrated_health_db):
    """migration 版本匹配时健康检查通过。"""
    async def _run():
        engine = create_async_engine(DSN)
        await check_memory_database_health(engine, "0002")
        await engine.dispose()

    asyncio.run(_run())


def test_health_check_fails_on_revision_mismatch(migrated_health_db):
    """migration 版本不匹配 → fail closed（不允许带错误 schema 启动）。"""
    async def _run():
        engine = create_async_engine(DSN)
        with pytest.raises(RuntimeError, match="版本不匹配"):
            await check_memory_database_health(engine, "9999")
        await engine.dispose()

    asyncio.run(_run())


def test_health_check_fails_on_missing_schema(migrated_health_db):
    """schema 缺失（表不存在）→ fail closed。"""
    async def _run():
        engine = create_async_engine(f"{BASE_URL}/nonexistent_db_xyz")
        with pytest.raises(RuntimeError):
            await check_memory_database_health(engine, "0002")
        await engine.dispose()

    asyncio.run(_run())
