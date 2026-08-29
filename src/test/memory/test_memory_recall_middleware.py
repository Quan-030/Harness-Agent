# src/test/memory/test_memory_recall_middleware.py
"""MemoryRecallMiddleware 集成测试（方案 18.6）。

覆盖：abefore_agent 预取（Profile + 启发式 Item 召回）、闲聊不召回、
revision 变化使 snapshot 失效、awrap_model_call 临时注入（SystemMessage
保留原 prompt metadata、HumanMessage 临时副本）、注入不写 state。
使用独立测试库 memory_v2_recall_test。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.models import (  # noqa: E402
    EntityType,
    MemoryKind,
    ProfilePatch,
    SourceType,
)
from agent.memory.recall import EntityRef  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    ActorContext,
    CreateMemoryCommand,
    MySQLMemoryRepository,
)
from agent.memory.recall import MemoryInvocationSnapshot  # noqa: E402
from agent.middlewares.memory_recall import MemoryRecallMiddleware  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_recall_test"
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


@pytest.fixture()
def repo_and_mw():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    repo = MySQLMemoryRepository(factory)
    yield repo, MemoryRecallMiddleware(repo)
    asyncio.run(engine.dispose())


# ============================================================
# 假 runtime / request / handler
# ============================================================

class FakeContext:
    def __init__(self, user_id: str = "u1", thread_id: str = "t1",
                 invocation_id: str = "inv-1", resume_id: str | None = None):
        self.user_id = user_id
        self.username = user_id
        self.thread_id = thread_id
        self.invocation_id = invocation_id
        self.resume_id = resume_id
        self.memory_snapshot = None


class FakeRuntime:
    def __init__(self, context: FakeContext):
        self.context = context


class FakeModelRequest:
    def __init__(self, runtime: FakeRuntime, system_message, messages):
        self.runtime = runtime
        self.system_message = system_message
        self.messages = messages
        self._override = None

    def override(self, **kwargs):
        self._override = kwargs
        return self


async def _seed_memory(repo, user_id: str, content: str) -> str:
    from agent.memory.models import compute_fingerprint, new_uuid7
    cmd = CreateMemoryCommand(
        user_id=user_id,
        memory_id=new_uuid7(),
        kind=MemoryKind.PROCUREMENT_CONSTRAINT,
        content=content,
        data={"constraint_name": "delivery_days_max", "value": 14},
        source_type=SourceType.USER_EXPLICIT,
        source_thread_id="t1",
        fingerprint=compute_fingerprint(
            MemoryKind.PROCUREMENT_CONSTRAINT, None, None, content,
            {"constraint_name": "delivery_days_max", "value": 14},
        ),
    )
    result = await repo.create_or_resolve_memory(
        cmd, ActorContext(actor_type="user", source_thread_id="t1")
    )
    return result.memory_id


# ============================================================
# abefore_agent 预取
# ============================================================

def test_abefore_agent_prefetches_profile_and_items(repo_and_mw):
    repo, mw = repo_and_mw

    async def _run():
        await _seed_memory(repo, "u-prefetch", "用户要求刹车片交期不超过 14 天")
        runtime = FakeRuntime(FakeContext(user_id="u-prefetch", invocation_id="inv-1"))
        state = {"messages": [HumanMessage(content="分析博世刹车片的交期", id="h-1")]}
        result = await mw.abefore_agent(state, runtime)
        assert result is None  # 不返回 state update（不污染 checkpoint）
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None
        assert snapshot.user_id == "u-prefetch"
        assert snapshot.profile.user_id == "u-prefetch"
        assert snapshot.latest_human_message_id == "h-1"
        # 采购关键词命中 → 召回 items
        assert len(snapshot.items) >= 1

    asyncio.run(_run())


def test_chitchat_only_loads_profile(repo_and_mw):
    """闲聊（无采购关键词）只加载 Profile，不召回 items（对齐点 3）。"""
    repo, mw = repo_and_mw

    async def _run():
        await _seed_memory(repo, "u-chat", "用户要求刹车片交期不超过 14 天")
        runtime = FakeRuntime(FakeContext(user_id="u-chat", invocation_id="inv-1"))
        state = {"messages": [HumanMessage(content="谢谢，再见", id="h-2")]}
        await mw.abefore_agent(state, runtime)
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None
        assert snapshot.profile.user_id == "u-chat"
        assert snapshot.items == []

    asyncio.run(_run())


def test_abefore_agent_degrades_on_prefetch_failure(repo_and_mw):
    """方案 5.8：召回失败降级为本轮无长期记忆，不阻断主流程。"""
    repo, mw = repo_and_mw

    async def _run():
        class FailingRepo:
            async def get_or_create_profile(self, user_id):
                raise RuntimeError("MySQL 连接断开")

            async def get_memory_revision(self, user_id):
                raise RuntimeError("MySQL 连接断开")

        failing_mw = MemoryRecallMiddleware(FailingRepo())
        runtime = FakeRuntime(FakeContext(user_id="u-degrade", invocation_id="inv-1"))
        state = {"messages": [HumanMessage(content="分析博世报价", id="h-9")]}
        result = await failing_mw.abefore_agent(state, runtime)
        assert result is None  # 不返回 state update
        assert runtime.context.memory_snapshot is None  # 降级：无记忆快照

    asyncio.run(_run())


def test_abefore_agent_without_human_message(repo_and_mw):
    repo, mw = repo_and_mw

    async def _run():
        runtime = FakeRuntime(FakeContext(user_id="u-none"))
        result = await mw.abefore_agent({"messages": []}, runtime)
        assert result is None
        assert runtime.context.memory_snapshot is None

    asyncio.run(_run())


# ============================================================
# snapshot 匹配（revision 变化 → 失效）
# ============================================================

def test_snapshot_invalidated_on_revision_change(repo_and_mw):
    repo, mw = repo_and_mw

    async def _run():
        user = "u-rev"
        await _seed_memory(repo, user, "旧约束内容")
        ctx = FakeContext(user_id=user, invocation_id="inv-1")
        runtime = FakeRuntime(ctx)
        await mw.abefore_agent(
            {"messages": [HumanMessage(content="分析交期", id="h-3")]}, runtime
        )
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None
        old_revision = snapshot.memory_revision

        # 写入新记忆 → revision +1 → 原 snapshot 失效
        await _seed_memory(repo, user, "新的约束内容")
        new_revision = await repo.get_memory_revision(user)
        assert new_revision > old_revision

        request = FakeModelRequest(
            runtime, SystemMessage(content="system prompt"),
            [HumanMessage(content="分析交期", id="h-3")],
        )
        matched = mw._snapshot_matches(
            snapshot, request, request.messages[0], new_revision
        )
        assert matched is False

    asyncio.run(_run())


# ============================================================
# P0 review：稳定编码 strict Enum / 实体分类 / Profile-Item 故障隔离
# ============================================================

def test_stable_order_code_builds_recall_intent():
    """订单号 PO-* 不触发 strict Enum ValidationError，生成 ORDER EntityRef。"""
    mw = MemoryRecallMiddleware(repository=None)  # 纯逻辑，无需 DB
    intent = mw._build_recall_intent("帮我查一下 PO-20260801 的订单状态")
    assert intent is not None
    assert intent.entity_refs == [
        EntityRef(entity_type=EntityType.ORDER, entity_id="PO-20260801")
    ]
    # 两路召回输入齐备：entity_refs（路径 A）+ query_text（路径 B）
    assert intent.query_text and "PO-20260801" in intent.query_text


def test_unclassifiable_code_does_not_fake_order():
    """ISO9001 不生成 ORDER EntityRef，但保留在 query_text 走全文召回。"""
    mw = MemoryRecallMiddleware(repository=None)  # 纯逻辑，无需 DB
    intent = mw._build_recall_intent("供应商要求符合 ISO9001 质量标准")
    assert intent is not None
    assert intent.entity_refs == []  # 不伪造实体类型
    assert intent.query_text and "ISO9001" in intent.query_text


def test_material_code_classified_as_material():
    """MAT-* 前缀 → MATERIAL 实体引用。"""
    from agent.middlewares.memory_recall import classify_entity_ref

    ref = classify_entity_ref("MAT-12345")
    assert ref is not None
    assert ref.entity_type == EntityType.MATERIAL
    assert ref.entity_id == "MAT-12345"
    # PO-* → ORDER；无法分类 → None
    assert classify_entity_ref("PO-1").entity_type == EntityType.ORDER
    assert classify_entity_ref("A1234") is None


def test_item_recall_failure_keeps_profile_only_snapshot(repo_and_mw):
    """P0 结构性修复：Item recall 失败 → profile-only snapshot（Profile 不得丢失）。"""
    repo, mw = repo_and_mw

    async def _run():
        user = "u-p0-sep"
        await repo.get_or_create_profile(user)
        await repo.patch_profile(
            user,
            ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "CNY"}]),
            expected_version=1,
            actor=ActorContext(actor_type="user", source_thread_id="t1"),
        )

        class ItemFailingRepo:
            """Profile 读取成功、Item recall 抛异常（隔离边界验证）。"""

            def __init__(self, inner):
                self._inner = inner

            async def get_or_create_profile(self, user_id):
                return await self._inner.get_or_create_profile(user_id)

            async def get_memory_revision(self, user_id):
                return await self._inner.get_memory_revision(user_id)

            async def search_memories(self, query):
                raise RuntimeError("FULLTEXT 故障")

        failing_mw = MemoryRecallMiddleware(ItemFailingRepo(repo))
        runtime = FakeRuntime(FakeContext(user_id=user, invocation_id="inv-1"))
        state = {"messages": [HumanMessage(content="查询 PO-20260801", id="h-p0")]}
        await failing_mw.abefore_agent(state, runtime)
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None  # 不再是 None（此前整个 snapshot 丢失）
        assert snapshot.profile.currency == "CNY"
        assert snapshot.items == []  # Item 降级为空

        # Profile 仍注入（Item recall failure != Profile injection failure）
        request = FakeModelRequest(
            runtime, SystemMessage(content="system"),
            [HumanMessage(content="查询 PO-20260801", id="h-p0")],
        )
        captured = {}

        async def handler(req):
            captured["req"] = req
            return "ok"

        await failing_mw.awrap_model_call(request, handler)
        sys_msg = captured["req"]._override["system_message"]
        assert "<user_profile_defaults>" in sys_msg.content
        assert "默认币种：CNY" in sys_msg.content

    asyncio.run(_run())


# ============================================================
# awrap_model_call 注入
# ============================================================

def test_awrap_injects_profile_and_items(repo_and_mw):
    repo, mw = repo_and_mw

    async def _run():
        user = "u-inject"
        await _seed_memory(repo, user, "用户要求刹车片交期不超过 14 天")
        # 预置 Profile 字段（验证 SystemMessage 默认值区块注入）
        await repo.get_or_create_profile(user)
        await repo.patch_profile(
            user,
            ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "CNY"}]),
            expected_version=1,
            actor=ActorContext(actor_type="user", source_thread_id="t1"),
        )
        ctx = FakeContext(user_id=user, invocation_id="inv-1")
        runtime = FakeRuntime(ctx)
        await mw.abefore_agent(
            {"messages": [HumanMessage(content="分析博世刹车片交期", id="h-4")]}, runtime
        )
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None and snapshot.items

        original_system = SystemMessage(
            content="你是 ERP 采购智能助手，固定系统规则",
            id="sys-1",
            name="main",
            additional_kwargs={"provider": "x"},
        )
        request = FakeModelRequest(
            runtime, original_system,
            [HumanMessage(content="分析博世刹车片交期", id="h-4")],
        )
        captured = {}

        async def handler(req):
            captured["req"] = req
            return "ok"

        await mw.awrap_model_call(request, handler)

        overridden = captured["req"]._override
        assert overridden is not None
        # Profile 区块追加到 SystemMessage 末尾，原 prompt 保留
        sys_msg = overridden["system_message"]
        assert sys_msg.content.startswith("你是 ERP 采购智能助手")
        assert "<user_profile_defaults>" in sys_msg.content
        # 原 SystemMessage metadata 保留（方案 18.6）
        assert sys_msg.id == "sys-1"
        assert sys_msg.name == "main"
        assert sys_msg.additional_kwargs == {"provider": "x"}
        # HumanMessage 临时副本：记忆块在前、问题在后
        human = overridden["messages"][0]
        assert "<retrieved_user_memory>" in human.content
        assert "<current_user_request>" in human.content
        assert human.content.index("交期不超过 14 天") < human.content.index("分析博世刹车片交期")
        assert human.id == "h-4"  # 原 message ID 保留

    asyncio.run(_run())


def test_awrap_no_items_no_augmentation(repo_and_mw):
    """无召回 items 时 HumanMessage 保持原样（不加空模板，方案 7.2）。"""
    repo, mw = repo_and_mw

    async def _run():
        ctx = FakeContext(user_id="u-plain", invocation_id="inv-1")
        runtime = FakeRuntime(ctx)
        await mw.abefore_agent(
            {"messages": [HumanMessage(content="谢谢", id="h-5")]}, runtime
        )
        snapshot = runtime.context.memory_snapshot
        assert snapshot is not None and snapshot.items == []

        request = FakeModelRequest(
            runtime, SystemMessage(content="system"), [HumanMessage(content="谢谢", id="h-5")]
        )
        captured = {}

        async def handler(req):
            captured["req"] = req
            return "ok"

        await mw.awrap_model_call(request, handler)
        human = captured["req"]._override["messages"][0]
        assert human.content == "谢谢"  # 未包装
        # Profile 仍注入（有默认值才注入）
        sys_msg = captured["req"]._override["system_message"]
        assert sys_msg.content == "system"  # 空 profile 不加空模板

    asyncio.run(_run())
