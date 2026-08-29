# src/test/memory/test_repository.py
"""MySQLMemoryRepository 集成测试（方案 17.3 / 18.1 / 20.2）。

覆盖：get_or_create_profile、patch_profile（乐观锁）、
create_or_resolve_memory（指纹去重/冲突 supersede/优先级）、
forget_memory（scope 校验）、list_memories（cursor 分页）、
memory_revision 递增、事务原子性。

使用独立测试库 memory_v2_repo_test，session 级 upgrade 一次。
"""
import asyncio
import os
import sys
from datetime import datetime, timezone
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
    MemoryStatus,
    ProfileListOp,
    ProfilePatch,
    SourceType,
    compute_fingerprint,
    new_uuid7,
)
from agent.memory.policies import MemoryPolicy  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    ActorContext,
    CreateMemoryCommand,
    MemoryListFilter,
    MemoryNotFound,
    MySQLMemoryRepository,
    ProfileVersionConflict,
)

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_repo_test"
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
def repo():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield MySQLMemoryRepository(factory)
    asyncio.run(engine.dispose())


@pytest.fixture()
def async_db():
    """直接 SQL 查询辅助（验证 revision/event 等）。"""
    engine = create_async_engine(DSN)

    async def _run(sql, **params):
        async with engine.begin() as conn:
            result = await conn.execute(text(sql), params)
            return result

    yield _run
    asyncio.run(engine.dispose())


def _command(
    user_id: str,
    *,
    kind: MemoryKind = MemoryKind.SUPPLIER_CONTEXT,
    content: str = "博世是长期合作供应商",
    data: dict | None = None,
    entity_type: EntityType | None = None,
    entity_id: str | None = None,
    source_type: SourceType = SourceType.USER_EXPLICIT,
) -> CreateMemoryCommand:
    data = data if data is not None else {"relationship": "长期合作"}
    fp = compute_fingerprint(kind, entity_type, entity_id, content, data)
    return CreateMemoryCommand(
        user_id=user_id,
        memory_id=new_uuid7(),
        kind=kind,
        content=content,
        data=data,
        entity_type=entity_type,
        entity_id=entity_id,
        source_type=source_type,
        source_thread_id="thread-1",
        fingerprint=fp,
    )


# ============================================================
# Profile
# ============================================================

def test_get_or_create_profile_is_idempotent(repo):
    user = f"repo-p-{new_uuid7()}"

    async def _run():
        p1 = await repo.get_or_create_profile(user)
        p2 = await repo.get_or_create_profile(user)
        return p1, p2

    p1, p2 = asyncio.run(_run())
    assert p1.user_id == user
    assert p1.version == 1
    assert p2.user_id == user
    assert p1.created_at == p2.created_at  # 幂等：同一行


def test_patch_profile_scalar_and_list_ops(repo):
    user = f"repo-patch-{new_uuid7()}"
    patch = ProfilePatch(
        scalar_ops=[
            {"field": "currency", "op": "set", "value": "CNY"},
            {"field": "delivery_days_max", "op": "set", "value": 14},
        ],
        list_ops=[
            ProfileListOp(field="blocked_suppliers", op="add", values=["博世"]),
            ProfileListOp(field="blocked_suppliers", op="add", values=["大陆"]),
        ],
    )

    async def _run():
        profile = await repo.get_or_create_profile(user)
        updated = await repo.patch_profile(user, patch, expected_version=profile.version, actor=ACTOR)
        return profile, updated

    _, updated = asyncio.run(_run())
    assert updated.currency == "CNY"
    assert updated.procurement_defaults.delivery_days_max == 14
    assert updated.procurement_defaults.blocked_suppliers == ["博世", "大陆"]
    assert updated.version == 2


def test_patch_profile_optimistic_lock(repo):
    """乐观锁：expected_version 不匹配 → 冲突；不匹配时 Profile 不变。"""
    user = f"repo-lock-{new_uuid7()}"
    patch = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "USD"}])

    async def _run():
        p1 = await repo.get_or_create_profile(user)
        await repo.patch_profile(user, patch, expected_version=p1.version, actor=ACTOR)
        # 再用旧的 expected_version → 冲突
        try:
            await repo.patch_profile(user, patch, expected_version=p1.version, actor=ACTOR)
        except ProfileVersionConflict:
            return True
        return False

    assert asyncio.run(_run()) is True


def test_patch_profile_clear_ops(repo):
    user = f"repo-clear-{new_uuid7()}"
    set_patch = ProfilePatch(
        scalar_ops=[
            {"field": "currency", "op": "set", "value": "CNY"},
            {"field": "language", "op": "set", "value": "zh"},
        ]
    )
    clear_patch = ProfilePatch(scalar_ops=[{"field": "currency", "op": "clear"}])

    async def _run():
        p = await repo.get_or_create_profile(user)
        p = await repo.patch_profile(user, set_patch, expected_version=p.version, actor=ACTOR)
        p = await repo.patch_profile(user, clear_patch, expected_version=p.version, actor=ACTOR)
        return p

    p = asyncio.run(_run())
    assert p.currency is None
    assert p.language == "zh"


# ============================================================
# Memory Item 写入
# ============================================================

def test_create_memory_and_revision_bumped(repo, async_db):
    user = f"repo-create-{new_uuid7()}"

    async def _run():
        result = await repo.create_or_resolve_memory(_command(user), ACTOR)
        revision = await repo.get_memory_revision(user)
        return result, revision

    result, revision = asyncio.run(_run())
    assert result.outcome == "created"
    assert revision == 1  # 写入后 revision +1


def test_duplicate_fingerprint_refreshes_not_inserts(repo, async_db):
    user = f"repo-dup-{new_uuid7()}"

    async def _run():
        cmd = _command(user)
        r1 = await repo.create_or_resolve_memory(cmd, ACTOR)
        r2 = await repo.create_or_resolve_memory(cmd, ACTOR)
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return r1, r2, page

    r1, r2, page = asyncio.run(_run())
    assert r1.outcome == "created"
    assert r2.outcome == "duplicate"
    assert len(page.items) == 1  # 不新增行


def test_conflict_supersedes_old_item(repo):
    """同 conflict_key（kind+entity+note_type）内容不同 → 新 Item 写入，旧 Item 标 superseded。"""
    user = f"repo-sup-{new_uuid7()}"
    old = _command(
        user,
        kind=MemoryKind.SUPPLIER_CONTEXT,
        content="博世是长期合作供应商",
        data={"note_type": "relationship", "relationship": "长期合作"},
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
    )
    new = _command(
        user,
        kind=MemoryKind.SUPPLIER_CONTEXT,
        content="博世合作关系已终止",
        data={"note_type": "relationship", "relationship": "已终止合作"},
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
    )

    async def _run():
        r1 = await repo.create_or_resolve_memory(old, ACTOR)
        r2 = await repo.create_or_resolve_memory(new, ACTOR)
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return r1, r2, page

    r1, r2, page = asyncio.run(_run())
    assert r1.outcome == "created"
    assert r2.outcome == "superseded"
    assert r2.superseded_memory_ids == [r1.memory_id]
    assert len(page.items) == 1  # 默认列表只显示 active：只有新条目
    assert page.items[0].content == "博世合作关系已终止"


def test_low_priority_does_not_override_high(repo):
    """低优先级（model_inferred）不得覆盖高优先级（user_explicit）旧事实。"""
    user = f"repo-prio-{new_uuid7()}"
    high = _command(
        user,
        kind=MemoryKind.PROCUREMENT_CONSTRAINT,
        content="交期上限 14 天",
        data={"constraint_name": "delivery_days_max", "value": 14},
        entity_type=EntityType.MATERIAL,
        entity_id="m-1",
    )
    low = _command(
        user,
        kind=MemoryKind.PROCUREMENT_CONSTRAINT,
        content="交期上限 30 天",
        data={"constraint_name": "delivery_days_max", "value": 30},
        entity_type=EntityType.MATERIAL,
        entity_id="m-1",
        source_type=SourceType.MODEL_INFERRED,
    )

    async def _run():
        r1 = await repo.create_or_resolve_memory(high, ACTOR)
        r2 = await repo.create_or_resolve_memory(low, ACTOR)
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return r1, r2, page

    r1, r2, page = asyncio.run(_run())
    # 低优先级不得覆盖：r2 被拒绝（outcome=conflict，不写入），旧条目保持 active
    assert r2.outcome == "conflict"
    assert len(page.items) == 1
    assert page.items[0].content == "交期上限 14 天"


def test_task_outcome_never_conflicts(repo):
    """task_outcome 默认不冲突：同实体不同结果各自保存。"""
    user = f"repo-task-{new_uuid7()}"

    async def _run():
        r1 = await repo.create_or_resolve_memory(
            _command(
                user,
                kind=MemoryKind.TASK_OUTCOME,
                content="询价博世结果",
                data={"task_type": "询价", "result_status": "succeeded"},
                entity_type=EntityType.SUPPLIER,
                entity_id="sup-1",
            ),
            ACTOR,
        )
        r2 = await repo.create_or_resolve_memory(
            _command(
                user,
                kind=MemoryKind.TASK_OUTCOME,
                content="询价博世结果",
                data={"task_type": "询价", "result_status": "failed"},
                entity_type=EntityType.SUPPLIER,
                entity_id="sup-1",
            ),
            ACTOR,
        )
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return r1, r2, page

    r1, r2, page = asyncio.run(_run())
    assert r2.outcome == "created"
    assert len(page.items) == 2


# ============================================================
# forget
# ============================================================

def test_forget_memory(repo):
    user = f"repo-forget-{new_uuid7()}"

    async def _run():
        r = await repo.create_or_resolve_memory(_command(user), ACTOR)
        await repo.forget_memory(user, r.memory_id, "用户要求忘记", ACTOR)
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return page

    page = asyncio.run(_run())
    assert len(page.items) == 0  # forgotten 不再可召回


def test_forget_other_users_memory_rejected(repo):
    """scope 校验：不能 forget 其他用户的记忆。"""
    owner = f"repo-owner-{new_uuid7()}"
    other = f"repo-other-{new_uuid7()}"

    async def _run():
        r = await repo.create_or_resolve_memory(_command(owner), ACTOR)
        try:
            await repo.forget_memory(other, r.memory_id, "越权", ACTOR)
        except MemoryNotFound:
            return True
        return False

    assert asyncio.run(_run()) is True


# ============================================================
# 分页
# ============================================================

def test_list_memories_cursor_pagination(repo):
    user = f"repo-page-{new_uuid7()}"

    async def _run():
        # 用 task_outcome（默认不冲突）保证 5 条都独立保存
        for i in range(5):
            await repo.create_or_resolve_memory(
                _command(
                    user,
                    kind=MemoryKind.TASK_OUTCOME,
                    content=f"任务结果 {i}",
                    data={"task_type": "询价", "result_status": "succeeded"},
                ),
                ACTOR,
            )
        page1 = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=2)
        assert len(page1.items) == 2
        assert page1.has_more is True
        assert page1.next_cursor is not None

        page2 = await repo.list_memories(user, MemoryListFilter(), cursor=page1.next_cursor, limit=2)
        page3 = await repo.list_memories(user, MemoryListFilter(), cursor=page2.next_cursor, limit=2)
        return page1, page2, page3

    p1, p2, p3 = asyncio.run(_run())
    assert len(p2.items) == 2
    assert len(p3.items) == 1
    assert p3.has_more is False
    # 不重叠
    ids = [i.memory_id for page in (p1, p2, p3) for i in page.items]
    assert len(ids) == len(set(ids)) == 5


def test_list_filters_kind_and_status(repo):
    user = f"repo-filter-{new_uuid7()}"

    async def _run():
        r1 = await repo.create_or_resolve_memory(_command(user), ACTOR)
        await repo.create_or_resolve_memory(
            _command(
                user,
                kind=MemoryKind.TASK_OUTCOME,
                content="任务结果",
                data={"task_type": "询价", "result_status": "succeeded"},
            ),
            ACTOR,
        )
        await repo.forget_memory(user, r1.memory_id, None, ACTOR)
        active_kind = await repo.list_memories(
            user, MemoryListFilter(kind=MemoryKind.TASK_OUTCOME), cursor=None, limit=50
        )
        all_active = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        forgotten = await repo.list_memories(
            user, MemoryListFilter(status=MemoryStatus.FORGOTTEN), cursor=None, limit=50
        )
        return active_kind, all_active, forgotten

    active_kind, all_active, forgotten = asyncio.run(_run())
    assert all(i.kind == MemoryKind.TASK_OUTCOME for i in active_kind.items)
    assert len(all_active.items) == 1
    assert len(forgotten.items) == 1
    assert forgotten.items[0].status == MemoryStatus.FORGOTTEN


# ============================================================
# 并发与事务不变量（review B/C/D/J）
# ============================================================

def test_concurrent_patch_same_version_one_wins(repo):
    """并发两个 PATCH 同一 version（Repository 层，无重试）→ 恰好一个成功、一个冲突。"""
    user = f"repo-ccas-{new_uuid7()}"
    patch_a = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "USD"}])
    patch_b = ProfilePatch(scalar_ops=[{"field": "language", "op": "set", "value": "en"}])

    async def _run():
        profile = await repo.get_or_create_profile(user)
        outcomes = await asyncio.gather(
            repo.patch_profile(user, patch_a, profile.version, ACTOR),
            repo.patch_profile(user, patch_b, profile.version, ACTOR),
            return_exceptions=True,
        )
        return outcomes

    outcomes = asyncio.run(_run())
    results = [o for o in outcomes if not isinstance(o, Exception)]
    conflicts = [o for o in outcomes if isinstance(o, ProfileVersionConflict)]
    assert len(results) == 1
    assert len(conflicts) == 1


def test_concurrent_patch_different_fields_both_survive(repo):
    """Service 层（带重试）并发更新不同字段 → 两个字段最终都保留（review B）。"""
    user = f"repo-cdiff-{new_uuid7()}"
    patch_a = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "USD"}])
    patch_b = ProfilePatch(scalar_ops=[{"field": "language", "op": "set", "value": "en"}])

    from agent.memory.service import MemoryService

    service = MemoryService(repo, MemoryPolicy())

    async def _run():
        await service.get_profile(user)
        outcomes = await asyncio.gather(
            service.update_preference(user, patch_a, ACTOR),
            service.update_preference(user, patch_b, ACTOR),
            return_exceptions=True,
        )
        assert not [o for o in outcomes if isinstance(o, Exception)]
        final = await service.get_profile(user)
        return final

    final = asyncio.run(_run())
    assert final.currency == "USD"
    assert final.language == "en"


def test_concurrent_patch_same_field_one_conflicts(repo):
    """Service 层并发更新同一字段 → 一个成功、一个最终冲突（禁止 silent last-write-wins）。"""
    user = f"repo-csame-{new_uuid7()}"
    patch_a = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "USD"}])
    patch_b = ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "EUR"}])

    from agent.memory.service import MemoryService

    service = MemoryService(repo, MemoryPolicy())

    async def _run():
        await service.get_profile(user)
        outcomes = await asyncio.gather(
            service.update_preference(user, patch_a, ACTOR),
            service.update_preference(user, patch_b, ACTOR),
            return_exceptions=True,
        )
        return outcomes

    outcomes = asyncio.run(_run())
    results = [o for o in outcomes if not isinstance(o, Exception)]
    conflicts = [o for o in outcomes if isinstance(o, ProfileVersionConflict)]
    assert len(results) == 1
    assert len(conflicts) == 1


def test_concurrent_same_fingerprint_created_and_duplicate(repo):
    """并发同 fingerprint 两写 → 结果集恰为 {created, duplicate}（review #2/D）。"""
    user = f"repo-cfp-{new_uuid7()}"

    async def _run():
        cmd = _command(user, content="并发同指纹记忆")
        outcomes = await asyncio.gather(
            repo.create_or_resolve_memory(cmd, ACTOR),
            repo.create_or_resolve_memory(cmd, ACTOR),
            return_exceptions=True,
        )
        return outcomes

    outcomes = asyncio.run(_run())
    assert not [o for o in outcomes if isinstance(o, Exception)]
    results = sorted(o.outcome for o in outcomes if not isinstance(o, Exception))
    assert results == ["created", "duplicate"]


def test_concurrent_same_conflict_key_single_active(repo):
    """并发同 conflict key 不同内容两写 → 仅一条 active（review #2）。"""
    user = f"repo-cconf-{new_uuid7()}"
    cmd_a = _command(
        user,
        kind=MemoryKind.SUPPLIER_CONTEXT,
        content="博世合作关系 A",
        data={"note_type": "relationship", "relationship": "A"},
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-9",
    )
    cmd_b = _command(
        user,
        kind=MemoryKind.SUPPLIER_CONTEXT,
        content="博世合作关系 B",
        data={"note_type": "relationship", "relationship": "B"},
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-9",
    )

    async def _run():
        outcomes = await asyncio.gather(
            repo.create_or_resolve_memory(cmd_a, ACTOR),
            repo.create_or_resolve_memory(cmd_b, ACTOR),
            return_exceptions=True,
        )
        assert not [o for o in outcomes if isinstance(o, Exception)]
        page = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=50)
        return page

    page = asyncio.run(_run())
    assert len(page.items) == 1  # 同 conflict key 只允许一条 active


def test_concurrent_get_or_create_profile_same_row(repo):
    """并发 get_or_create_profile（全新用户）→ 都成功且同一行（review #3）。"""
    user = f"repo-cgoc-{new_uuid7()}"

    async def _run():
        outcomes = await asyncio.gather(
            repo.get_or_create_profile(user),
            repo.get_or_create_profile(user),
            return_exceptions=True,
        )
        return outcomes

    outcomes = asyncio.run(_run())
    assert not [o for o in outcomes if isinstance(o, Exception)]
    assert outcomes[0].user_id == outcomes[1].user_id
    assert outcomes[0].created_at == outcomes[1].created_at  # 同一行


def test_duplicate_refresh_bumps_revision(repo):
    """duplicate 刷新 updated_at 同步递增 memory_revision（review J）。"""
    user = f"repo-cdup-{new_uuid7()}"

    async def _run():
        cmd = _command(user)
        await repo.create_or_resolve_memory(cmd, ACTOR)
        r1 = await repo.get_memory_revision(user)
        await repo.create_or_resolve_memory(cmd, ACTOR)  # duplicate
        r2 = await repo.get_memory_revision(user)
        return r1, r2

    r1, r2 = asyncio.run(_run())
    assert r1 == 1
    assert r2 == 2


def test_cursor_pagination_same_timestamp_no_gap(repo, async_db):
    """同 updated_at 多条记录跨页分页 → 不漏不重（review #6 keyset tie-breaker）。"""
    user = f"repo-cts-{new_uuid7()}"

    async def _run():
        created = []
        for i in range(5):
            r = await repo.create_or_resolve_memory(
                _command(
                    user,
                    kind=MemoryKind.TASK_OUTCOME,
                    content=f"同时间任务 {i}",
                    data={"task_type": "询价", "result_status": "succeeded"},
                ),
                ACTOR,
            )
            created.append(r.memory_id)
        # 将所有条目 updated_at 强制设为同一时间戳（模拟同毫秒写入）
        await async_db(
            "UPDATE memory_items SET updated_at = '2026-08-08 12:00:00.000000' WHERE user_id=:u",
            u=user,
        )
        p1 = await repo.list_memories(user, MemoryListFilter(), cursor=None, limit=2)
        p2 = await repo.list_memories(user, MemoryListFilter(), cursor=p1.next_cursor, limit=2)
        p3 = await repo.list_memories(user, MemoryListFilter(), cursor=p2.next_cursor, limit=2)
        return p1, p2, p3, created

    p1, p2, p3, created = asyncio.run(_run())
    ids = [i.memory_id for page in (p1, p2, p3) for i in page.items]
    assert len(ids) == 5
    assert len(set(ids)) == 5  # 不重叠
    assert set(ids) == set(created)  # 不漏
    assert p3.has_more is False
