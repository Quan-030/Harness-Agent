# src/test/memory/test_service.py
"""MemoryService 集成测试（方案 18.1 / 20.1 / 20.3）。

覆盖：update_preference（版本自动重试/冲突传播）、remember（TTL/长期/敏感拒绝）、
forget_memory（精确 ID / 模糊 query 0-1-多语义）、list_memories。
"""
import asyncio
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.models import (  # noqa: E402
    EntityType,
    MemoryKind,
    ProfileListOp,
    ProfilePatch,
    SourceType,
    new_uuid7,
)
from agent.memory.policies import MemoryPolicy, SensitiveContentError  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    ActorContext,
    InvalidMemoryData,
    MemoryListFilter,
    MySQLMemoryRepository,
    ProfileVersionConflict,
)
from agent.memory.service import MemoryService  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_service_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

ACTOR = ActorContext(actor_type="user", source_thread_id="thread-1")


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    """session 级：创建测试库并 upgrade 一次，teardown 时删除。"""
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


@pytest.fixture()
def service():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    repo = MySQLMemoryRepository(factory)
    yield MemoryService(repo, MemoryPolicy())
    asyncio.run(engine.dispose())


# ============================================================
# update_preference
# ============================================================

def test_update_preference_auto_reads_version(service):
    """expected_version 缺失时自动读版本并更新。"""
    user = f"svc-p-{new_uuid7()}"
    patch = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "CNY"}])

    async def _run():
        profile = await service.update_preference(user, patch, ACTOR, expected_version=None)
        return profile

    profile = asyncio.run(_run())
    assert profile.currency == "CNY"


def test_update_preference_conflict_propagates(service):
    """显式提供过期版本 → 冲突传播（API 层据此返回 409）。"""
    user = f"svc-conflict-{new_uuid7()}"
    patch = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "CNY"}])

    async def _run():
        await service.update_preference(user, patch, ACTOR, expected_version=None)
        try:
            # 用过期版本（1）再更新 → 冲突
            await service.update_preference(user, patch, ACTOR, expected_version=1)
        except ProfileVersionConflict:
            return True
        return False

    assert asyncio.run(_run()) is True


def test_update_preference_list_ops(service):
    user = f"svc-list-{new_uuid7()}"
    patch = ProfilePatch(
        list_ops=[ProfileListOp(field="quality_standards", op="add", values=["ISO9001"])]
    )

    async def _run():
        profile = await service.update_preference(user, patch, ACTOR)
        return profile

    profile = asyncio.run(_run())
    assert profile.procurement_defaults.quality_standards == ["ISO9001"]


# ============================================================
# remember（TTL / 敏感 / 长期）
# ============================================================

def test_remember_applies_default_ttl(service):
    """supplier_context 默认 TTL 180 天。"""
    user = f"svc-ttl-{new_uuid7()}"

    async def _run():
        result = await service.remember(
            user, ACTOR,
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="博世是长期合作供应商",
            data={"relationship": "长期合作"},
            entity_type=EntityType.SUPPLIER,
            entity_id="sup-1",
        )
        page = await service.list_memories(user, MemoryListFilter(), limit=10)
        return result, page

    result, page = asyncio.run(_run())
    assert result.outcome == "created"
    item = page.items[0]
    assert item.expires_at is not None
    # expires_at 与 created_at 分别由 Service/Repository 生成（微秒级差），允许亚秒容差
    ttl = item.expires_at - item.created_at
    assert timedelta(days=180) - timedelta(seconds=1) <= ttl <= timedelta(days=180)


def test_remember_long_term_no_expiry(service):
    user = f"svc-long-{new_uuid7()}"

    async def _run():
        await service.remember(
            user, ACTOR,
            kind=MemoryKind.TASK_OUTCOME,
            content="用户要求长期记住博世合作",
            data={"task_type": "合作", "result_status": "succeeded"},
            long_term=True,
        )
        page = await service.list_memories(user, MemoryListFilter(), limit=10)
        return page

    page = asyncio.run(_run())
    assert page.items[0].expires_at is None


def test_remember_rejects_sensitive_content(service):
    """敏感内容整条拒绝，不写入。"""
    user = f"svc-sensitive-{new_uuid7()}"

    async def _run():
        try:
            await service.remember(
                user, ACTOR,
                kind=MemoryKind.USER_FEEDBACK,
                content="用户说密码是 abc123",
                data={"target_type": "workflow", "feedback_type": "negative"},
            )
        except SensitiveContentError:
            page = await service.list_memories(user, MemoryListFilter(), limit=10)
            return len(page.items)
        return -1

    assert asyncio.run(_run()) == 0


# ============================================================
# forget_memory（方案 20.1 语义）
# ============================================================

def test_forget_by_exact_id(service):
    user = f"svc-fid-{new_uuid7()}"

    async def _run():
        result = await service.remember(
            user, ACTOR,
            kind=MemoryKind.TASK_OUTCOME,
            content="任务结果 1",
            data={"task_type": "询价", "result_status": "succeeded"},
        )
        forget = await service.forget_memory(user, ACTOR, memory_id=result.memory_id)
        page = await service.list_memories(user, MemoryListFilter(), limit=10)
        return forget, page

    forget, page = asyncio.run(_run())
    assert forget.outcome == "forgotten"
    assert len(page.items) == 0


def test_forget_by_query_single_match(service):
    user = f"svc-fq1-{new_uuid7()}"

    async def _run():
        await service.remember(
            user, ACTOR,
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="博世报价分析结果",
            data={"note_type": "other", "relationship": None},
        )
        forget = await service.forget_memory(user, ACTOR, query="博世")
        return forget

    forget = asyncio.run(_run())
    assert forget.outcome == "forgotten"


def test_forget_by_query_no_match(service):
    user = f"svc-fq0-{new_uuid7()}"

    async def _run():
        return await service.forget_memory(user, ACTOR, query="不存在的记忆")

    assert asyncio.run(_run()).outcome == "not_found"


def test_forget_by_query_ambiguous(service):
    """多条命中 → ambiguous，展示候选不执行。"""
    user = f"svc-fqm-{new_uuid7()}"

    async def _run():
        for i in range(2):
            await service.remember(
                user, ACTOR,
                kind=MemoryKind.TASK_OUTCOME,
                content=f"博世询价结果 {i}",
                data={"task_type": "询价", "result_status": "succeeded"},
            )
        forget = await service.forget_memory(user, ACTOR, query="博世")
        page = await service.list_memories(user, MemoryListFilter(), limit=10)
        return forget, page

    forget, page = asyncio.run(_run())
    assert forget.outcome == "ambiguous"
    assert len(forget.candidates) == 2
    assert len(page.items) == 2  # 未删除


# ============================================================
# model_inferred 拦截（review 评论二）
# ============================================================

def test_remember_rejects_model_inferred(service):
    """v1 禁止 model_inferred 自动入库：抛异常且不入库。"""
    user = f"svc-inf-{new_uuid7()}"

    async def _run():
        try:
            await service.remember(
                user,
                ActorContext(actor_type="user", source_thread_id="t1"),
                kind=MemoryKind.SUPPLIER_CONTEXT,
                content="推断的内容",
                source_type=SourceType.MODEL_INFERRED,
            )
        except ValueError as exc:
            assert "model_inferred" in str(exc)
            page = await service.list_memories(user, MemoryListFilter(), limit=10)
            return len(page.items)
        return -1

    assert asyncio.run(_run()) == 0


def test_remember_accepts_user_explicit_and_tool_verified(service):
    """user_explicit / tool_verified 两类合法来源正常持久化。"""
    user = f"svc-src-{new_uuid7()}"

    async def _run():
        r1 = await service.remember(
            user,
            ActorContext(actor_type="user", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="用户明确要求记住",
            data={"task_type": "合作", "result_status": "succeeded"},
        )
        r2 = await service.remember(
            user,
            ActorContext(actor_type="system", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="工具验证的订单结果",
            data={"task_type": "下单", "result_status": "succeeded"},
            source_type=SourceType.TOOL_VERIFIED,
        )
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1.outcome == "created"
    assert r2.outcome == "created"


def test_fuzzy_forget_scan_beyond_first_page_is_ambiguous(service):
    """review 3.1：101+ 条记忆、第一页 1 个 match、后续页还有 match
    → 必须 ambiguous，任何 Item 都不能被 forgotten。"""
    user = f"svc-scan-{new_uuid7()}"

    async def _run():
        # 150 条 active 记忆（task_outcome 不冲突），其中 3 条含"博世"
        for i in range(147):
            await service.remember(
                user,
                ActorContext(actor_type="user", source_thread_id="t1"),
                kind=MemoryKind.TASK_OUTCOME,
                content=f"普通任务记忆 {i}",
                data={"task_type": "询价", "result_status": "succeeded"},
            )
        for i in range(3):
            await service.remember(
                user,
                ActorContext(actor_type="user", source_thread_id="t1"),
                kind=MemoryKind.TASK_OUTCOME,
                content=f"博世相关任务 {i}",
                data={"task_type": "询价", "result_status": "succeeded"},
            )
        result = await service.forget_memory(user, ActorContext(actor_type="user", source_thread_id="t1"), query="博世")
        page = await service.list_memories(user, MemoryListFilter(), limit=10)
        return result, page

    result, page = asyncio.run(_run())
    assert result.outcome == "ambiguous"
    assert result.match_count == 3
    assert result.has_more is False
    assert len(result.candidates) == 3
    # 任何 Item 都未被删除
    assert len(page.items) == 10  # 列表仍满（未被 forget）


def test_fuzzy_forget_empty_query_rejected(service):
    """review 3.1：空 query 拒绝（"" 会命中所有记忆，单条记忆会被误删）。"""
    user = f"svc-empty-{new_uuid7()}"

    async def _run():
        try:
            await service.forget_memory(user, ActorContext(actor_type="user", source_thread_id="t1"), query="")
        except InvalidMemoryData:
            return True
        return False

    assert asyncio.run(_run()) is True


def test_fuzzy_forget_single_match_executes(service):
    """单条匹配仍正常执行 forget（review 3.1 语义 1 result → forget）。"""
    user = f"svc-one-{new_uuid7()}"

    async def _run():
        await service.remember(
            user,
            ActorContext(actor_type="user", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="唯一的博世任务",
            data={"task_type": "询价", "result_status": "succeeded"},
        )
        result = await service.forget_memory(user, ActorContext(actor_type="user", source_thread_id="t1"), query="博世")
        return result

    result = asyncio.run(_run())
    assert result.outcome == "forgotten"
