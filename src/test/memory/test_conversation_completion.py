# src/test/memory/test_conversation_completion.py
"""ConversationCompletionService 测试（方案 6.1 / 19.2）。

覆盖：展示消息 ID 提取、入队幂等、ID 缺失不创建不完整 Job。
使用独立测试库 memory_v2_completion_test。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from api_view.conversation_completion import (  # noqa: E402
    ConversationCompletionService,
    extract_ids_from_display_messages,
)
from agent.memory.repository import MySQLMemoryRepository  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_completion_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    """session 级：建库 + upgrade；teardown 时删除。"""
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


@pytest.fixture(autouse=True)
def clean_jobs():
    yield
    async def _clean():
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM memory_jobs"))
        await engine.dispose()
    asyncio.run(_clean())


@pytest.fixture()
def completion():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield ConversationCompletionService(MySQLMemoryRepository(factory))
    asyncio.run(engine.dispose())


# ============================================================
# 展示消息 ID 提取（方案 6.1）
# ============================================================

def test_extract_ids_from_display_messages():
    messages = [
        {"id": "user-1", "role": "user", "content": "分析博世报价"},
        {"id": "tool-1", "role": "tool", "tool_status": "done"},
        {"id": "assistant-1", "role": "assistant", "content": "正在分析"},
        {"id": "assistant-empty", "role": "assistant", "content": ""},  # 空内容忽略
    ]
    user_id, assistant_id = extract_ids_from_display_messages(messages)
    assert user_id == "user-1"
    assert assistant_id == "assistant-1"


def test_extract_ids_returns_none_when_missing():
    user_id, assistant_id = extract_ids_from_display_messages(
        [{"id": "tool-1", "role": "tool"}]
    )
    assert user_id is None
    assert assistant_id is None


# ============================================================
# 入队（方案 6.1 / 19.2）
# ============================================================

def test_enqueue_extract_creates_job(completion):
    async def _run():
        job_id = await completion.enqueue_extract(
            user_id="cc-1", thread_id="t1",
            user_message_id="user-1", assistant_message_id="assistant-1",
            checkpoint_id="cp-1",
        )
        job_id_2 = await completion.enqueue_extract(
            user_id="cc-1", thread_id="t1",
            user_message_id="user-1", assistant_message_id="assistant-1",
            checkpoint_id="cp-CHANGED",  # checkpoint 变化不影响幂等键
        )
        return job_id, job_id_2

    j1, j2 = asyncio.run(_run())
    assert j1 is not None
    assert j1 == j2  # 幂等：重复入队返回同一 job


def test_enqueue_extract_rejects_incomplete_ids(completion):
    async def _run():
        job_id = await completion.enqueue_extract(
            user_id="cc-2", thread_id="t1",
            user_message_id="user-1", assistant_message_id="",
            checkpoint_id="cp-1",
        )
        return job_id

    assert asyncio.run(_run()) is None  # 不创建不完整 Job


def test_enqueue_extract_writes_payload(completion):
    async def _run():
        await completion.enqueue_extract(
            user_id="cc-3", thread_id="t1",
            user_message_id="user-1", assistant_message_id="assistant-1",
            checkpoint_id="cp-1",
        )
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT job_type, payload FROM memory_jobs "
                        "WHERE user_id=:u"
                    ),
                    {"u": "cc-3"},
                )
            ).first()
        await engine.dispose()
        return row

    row = asyncio.run(_run())
    import json as _json
    payload = _json.loads(row.payload) if isinstance(row.payload, str) else row.payload
    assert row.job_type == "extract_memory"
    assert payload["user_message_id"] == "user-1"
    assert payload["assistant_message_id"] == "assistant-1"
    assert payload["checkpoint_id"] == "cp-1"
