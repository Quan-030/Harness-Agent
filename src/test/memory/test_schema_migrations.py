# src/test/memory/test_schema_migrations.py
"""MySQL schema migration 与约束集成测试（方案 17 节 + 21.4 DoD）。

- Alembic 可从空库 upgrade，可重复执行，downgrade/前滚恢复可验证
- 数据库 CHECK / UNIQUE 拒绝非法枚举与重复数据（Pydantic 之外的兜底层）
- active_fingerprint 生成列语义
- FULLTEXT ngram 中文检索真实可用

使用独立测试库 memory_v2_test（不触碰开发库 memory_v2），session 级 upgrade 一次。
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.models import new_uuid7  # noqa: E402
from agent.memory.orm_models import MemoryItemORM  # noqa: E402

# 与仓库现有测试同风格：默认本地凭据，可用环境变量覆盖
BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

NOW = datetime(2026, 8, 8, 0, 0, 0, tzinfo=timezone.utc)


def _alembic_config() -> Config:
    cfg = Config(ALEMBIC_INI)
    os.environ["MEMORY_MYSQL_DSN"] = DSN
    return cfg


async def _create_db() -> None:
    engine = create_async_engine(BASE_URL)
    async with engine.begin() as conn:
        await conn.execute(
            text(f"CREATE DATABASE IF NOT EXISTS {TEST_DB} CHARACTER SET utf8mb4")
        )
    await engine.dispose()


async def _drop_db() -> None:
    engine = create_async_engine(BASE_URL)
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB}"))
    await engine.dispose()


async def _check_tables() -> None:
    engine = create_async_engine(DSN)
    async with engine.begin() as conn:
        rows = (
            await conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = :db ORDER BY table_name"
                ),
                {"db": TEST_DB},
            )
        ).all()
        tables = {r[0] for r in rows}
    await engine.dispose()
    assert {
        "memory_profiles",
        "memory_items",
        "memory_events",
        "memory_jobs",
        "memory_user_state",
        "alembic_version",
    } <= tables


def _upgrade_and_check() -> None:
    """同步调用 alembic（env.py 内部自带 asyncio.run），表检查单独包 asyncio.run。"""
    command.upgrade(_alembic_config(), "head")
    asyncio.run(_check_tables())


# ============================================================
# Migration 生命周期
# ============================================================

def test_upgrade_from_empty_db():
    """Alembic 可从空库 upgrade 到 head。"""
    asyncio.run(_create_db())
    try:
        _upgrade_and_check()
    finally:
        asyncio.run(_drop_db())


def test_upgrade_is_idempotent():
    """upgrade 到 head 后重复执行不损坏数据。"""
    asyncio.run(_create_db())
    try:
        _upgrade_and_check()
        command.upgrade(_alembic_config(), "head")  # 重复执行
        command.upgrade(_alembic_config(), "head")  # 再重复
        _upgrade_and_check()
    finally:
        asyncio.run(_drop_db())


def test_downgrade_and_reupgrade():
    """downgrade 到 base 后重新 upgrade，schema 仍完整（前滚恢复）。"""
    asyncio.run(_create_db())
    try:
        _upgrade_and_check()
        command.downgrade(_alembic_config(), "base")
        _upgrade_and_check()
    finally:
        asyncio.run(_drop_db())


# ============================================================
# 数据约束（共享 session 级已升级库）
# ============================================================

@pytest.fixture(scope="session", autouse=True)
def migrated_session_db():
    """session 级：创建测试库并 upgrade 一次，teardown 时删除。"""
    asyncio.run(_create_db())
    _upgrade_and_check()
    yield
    asyncio.run(_drop_db())


@pytest.fixture()
def db_session():
    engine = create_async_engine(DSN)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async def _run(fn):
        async with session_factory() as session:
            return await fn(session)

    yield _run
    asyncio.run(engine.dispose())


def _item_row(user_id: str, *, kind: str = "supplier_context", status: str = "active",
              fingerprint: str | None = None, content: str = "博世是长期合作供应商"):
    return {
        "memory_id": new_uuid7(),
        "user_id": user_id,
        "kind": kind,
        "content": content,
        "data": {"relationship": "长期合作"},
        "source_type": "user_explicit",
        "source_thread_id": "thread-1",
        "status": status,
        "fingerprint": fingerprint or "f" * 64,
        "created_at": NOW.replace(tzinfo=None),
        "updated_at": NOW.replace(tzinfo=None),
    }


def test_check_rejects_invalid_enum(db_session):
    """数据库 CHECK 拒绝非法 kind（Pydantic 之外的兜底）。"""
    async def _run(session):
        with pytest.raises(IntegrityError):
            session.add(MemoryItemORM(**_item_row("u-bad-kind", kind="bogus_kind")))
            await session.commit()
        await session.rollback()
    db_session(_run)


def test_profile_check_rejects_invalid_enum(db_session):
    """review #3：memory_profiles 的 output_format/chart_type CHECK 拒绝非法枚举。"""
    async def _run(session):
        user = f"u-chk-{new_uuid7()}"
        await session.execute(
            text(
                "INSERT INTO memory_profiles "
                "(user_id, procurement_defaults, created_at, updated_at) "
                "VALUES (:u, JSON_OBJECT(), :t, :t)"
            ),
            {"u": user, "t": NOW.replace(tzinfo=None)},
        )
        await session.commit()
        # 非法 output_format
        with pytest.raises(IntegrityError):
            await session.execute(
                text("UPDATE memory_profiles SET output_format='excel' WHERE user_id=:u"),
                {"u": user},
            )
        await session.rollback()
        # 非法 chart_type
        with pytest.raises(IntegrityError):
            await session.execute(
                text("UPDATE memory_profiles SET chart_type='scatter' WHERE user_id=:u"),
                {"u": user},
            )
        await session.rollback()
        # 合法枚举不受影响
        await session.execute(
            text("UPDATE memory_profiles SET output_format='table', chart_type='bar' WHERE user_id=:u"),
            {"u": user},
        )
        await session.commit()
    db_session(_run)


def test_char_column_types(db_session):
    """review #4：UUID/fingerprint/currency 列使用 CHAR 而非 VARCHAR。"""
    async def _run(session):
        rows = (
            await session.execute(
                text(
                    "SELECT table_name, column_name, data_type, character_maximum_length "
                    "FROM information_schema.columns "
                    "WHERE table_schema = :db AND column_name IN "
                    "('memory_id','event_id','job_id','fingerprint','currency')"
                ),
                {"db": TEST_DB},
            )
        ).all()
        by_name = {(r.table_name, r.column_name): r for r in rows}
        assert by_name[("memory_items", "memory_id")].data_type == "char"
        assert by_name[("memory_items", "memory_id")].character_maximum_length == 36
        assert by_name[("memory_events", "event_id")].data_type == "char"
        assert by_name[("memory_jobs", "job_id")].data_type == "char"
        assert by_name[("memory_items", "fingerprint")].data_type == "char"
        assert by_name[("memory_items", "fingerprint")].character_maximum_length == 64
        assert by_name[("memory_profiles", "currency")].data_type == "char"
        assert by_name[("memory_profiles", "currency")].character_maximum_length == 3
    db_session(_run)


def test_unique_active_fingerprint(db_session):
    """UNIQUE(user_id, active_fingerprint)：同一用户同指纹的 active 记忆只允许一条。"""
    user = f"u-uniq-{new_uuid7()}"
    fp = "a" * 64

    async def _run(session):
        session.add(MemoryItemORM(**_item_row(user, fingerprint=fp)))
        await session.commit()
        # 第二条 active 同指纹 → 唯一约束冲突
        with pytest.raises(IntegrityError):
            session.add(MemoryItemORM(**_item_row(user, fingerprint=fp)))
            await session.commit()
        await session.rollback()
    db_session(_run)


def test_superseded_allows_same_fingerprint(db_session):
    """旧条目 superseded 后，同一指纹可再次写入（允许用户重新声明相同事实）。"""
    user = f"u-sup-{new_uuid7()}"
    fp = "b" * 64

    async def _run(session):
        session.add(MemoryItemORM(**_item_row(user, fingerprint=fp)))
        await session.commit()
        # 第一条 superseded
        await session.execute(
            text("UPDATE memory_items SET status='superseded' WHERE user_id=:u"),
            {"u": user},
        )
        await session.commit()
        # 同指纹新 active 条目可写入
        session.add(MemoryItemORM(**_item_row(user, fingerprint=fp)))
        await session.commit()
    db_session(_run)


def test_generated_column_follows_status(db_session):
    """active_fingerprint 生成列：active 时为 fingerprint，superseded 后为 NULL。"""
    user = f"u-gen-{new_uuid7()}"

    async def _run(session):
        session.add(MemoryItemORM(**_item_row(user)))
        await session.commit()
        row = (
            await session.execute(
                text("SELECT fingerprint, active_fingerprint FROM memory_items WHERE user_id=:u"),
                {"u": user},
            )
        ).one()
        assert row.active_fingerprint == row.fingerprint

        await session.execute(
            text("UPDATE memory_items SET status='superseded' WHERE user_id=:u"),
            {"u": user},
        )
        await session.commit()
        row = (
            await session.execute(
                text("SELECT active_fingerprint FROM memory_items WHERE user_id=:u"),
                {"u": user},
            )
        ).one()
        assert row.active_fingerprint is None
    db_session(_run)


def test_fulltext_ngram_chinese_search(db_session):
    """FULLTEXT ngram 对中文内容的真实检索（路径 B 的基础能力）。"""
    user = f"u-ft-{new_uuid7()}"

    async def _run(session):
        session.add(
            MemoryItemORM(
                **_item_row(
                    user,
                    kind="procurement_constraint",
                    content="用户要求刹车片采购交期不超过 14 天",
                    fingerprint="c" * 64,
                )
            )
        )
        await session.commit()
        row = (
            await session.execute(
                text(
                    "SELECT memory_id FROM memory_items "
                    "WHERE user_id=:u AND MATCH(content) AGAINST(:q IN NATURAL LANGUAGE MODE)"
                ),
                {"u": user, "q": "交期"},
            )
        ).first()
        assert row is not None
    db_session(_run)


def test_default_values_applied(db_session):
    """status/schema_version/version/attempts 等列使用数据库 DEFAULT。"""
    async def _run(session):
        user = f"u-def-{new_uuid7()}"
        await session.execute(
            text(
                "INSERT INTO memory_profiles "
                "(user_id, procurement_defaults, created_at, updated_at) "
                "VALUES (:u, JSON_OBJECT(), :t, :t)"
            ),
            {"u": user, "t": NOW.replace(tzinfo=None)},
        )
        await session.execute(
            text(
                "INSERT INTO memory_jobs "
                "(job_id, idempotency_key, user_id, thread_id, job_type, payload, "
                " available_at, created_at, updated_at) "
                "VALUES (:jid, :ik, :u, 't1', 'extract_memory', JSON_OBJECT(), :t, :t, :t)"
            ),
            {
                "jid": new_uuid7(),
                "ik": new_uuid7(),
                "u": user,
                "t": NOW.replace(tzinfo=None),
            },
        )
        await session.commit()
        profile = (
            await session.execute(
                text("SELECT schema_version, version FROM memory_profiles WHERE user_id=:u"),
                {"u": user},
            )
        ).one()
        job = (
            await session.execute(
                text("SELECT status, attempts FROM memory_jobs WHERE user_id=:u"),
                {"u": user},
            )
        ).one()
        assert profile.schema_version == 2
        assert profile.version == 1
        assert job.status == "pending"
        assert job.attempts == 0
    db_session(_run)
