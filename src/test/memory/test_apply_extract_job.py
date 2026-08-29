# src/test/memory/test_apply_extract_job.py
"""PR #15 review 修复验证测试（#1/#3/#4）。

覆盖：
- apply_extract_job 单事务 invariant：两种线性化顺序下旧 Job 都不复活记忆
- stale worker（lease 回收后旧 worker 被拒）
- 本轮区间提取（旧 thread 工具事件不被刷新）
- 引用消息缺失 → failed/retry（不 succeeded）
- processing lease 到期 → pending → 其他 worker 可重新 claim
使用独立测试库 memory_v2_apply_test。
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.models import MemoryKind, SourceType  # noqa: E402
from agent.memory.policies import MemoryPolicy  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    ActorContext,
    EnqueueJobCommand,
    MemoryListFilter,
    MySQLMemoryRepository,
)
from agent.memory.service import MemoryService  # noqa: E402
from agent.memory.worker import MemoryWorker, ReferencedMessageUnavailable  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_apply_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
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
def clean_data():
    yield
    async def _clean():
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM memory_jobs"))
            await conn.execute(text("DELETE FROM memory_items"))
            await conn.execute(text("DELETE FROM memory_profiles"))
            await conn.execute(text("DELETE FROM memory_user_state"))
        await engine.dispose()
    asyncio.run(_clean())


@pytest.fixture()
def repo():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield MySQLMemoryRepository(factory)
    asyncio.run(engine.dispose())


def _extract_command(user: str) -> EnqueueJobCommand:
    return EnqueueJobCommand(
        user_id=user,
        thread_id="t1",
        job_type="extract_memory",
        payload={
            "checkpoint_id": "cp-1",
            "user_message_id": "user-1",
            "assistant_message_id": "assistant-1",
            "extractor_version": "memory-v2.1",
            "memory_generation": 0,
            "replay_generation": 0,
        },
    )


def _service(repo) -> MemoryService:
    return MemoryService(repo, MemoryPolicy())


async def _active_item_count(repo, user: str) -> int:
    page = await repo.list_memories(user, MemoryListFilter(), None, 50)
    return len(page.items)


async def _job_status(repo, user: str) -> str:
    engine = create_async_engine(DSN)
    async with engine.begin() as conn:
        row = (
            await conn.execute(
                text("SELECT status FROM memory_jobs WHERE user_id=:u"),
                {"u": user},
            )
        ).first()
    await engine.dispose()
    return row[0] if row else None


async def _claim_and_apply(
    repo, job_id: str, expected_generation: int, commands, worker_id: str = "w1",
):
    """模拟 worker：claim（processing）→ apply_extract_job。"""
    claimed = await repo.claim_jobs(
        worker_id, limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
    )
    assert any(j.job_id == job_id for j in claimed)
    return await repo.apply_extract_job(
        job_id=job_id,
        worker_id=worker_id,
        user_id=claimed[0].user_id,
        expected_generation=expected_generation,
        commands=commands,
        actor=ActorContext(actor_type="system", source_thread_id="t1"),
    )


def _tool_verified_command(repo, user: str, content: str = "订单创建成功：PO-1"):
    return _service(repo).prepare_memory_command(
        user_id=user,
        actor=ActorContext(actor_type="system", source_thread_id="t1"),
        kind=MemoryKind.TASK_OUTCOME,
        content=content,
        data={"task_type": "订单创建", "result_status": "succeeded"},
        source_type=SourceType.TOOL_VERIFIED,
    )


# ============================================================
# #1: apply_extract_job 并发 invariant
# ============================================================

def test_apply_before_delete_all_no_resurrection(repo):
    """线性化顺序 1：worker apply 先提交 → delete-all 后清空。最终 active=0，job=succeeded。"""
    user = "apply-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        outcome = await _claim_and_apply(
            repo, (await repo.get_job_by_idempotency_key(_key(user))).job_id,
            0, [_tool_verified_command(repo, user)],
        )
        assert outcome == "succeeded"
        await repo.delete_all_memories(user, ActorContext(actor_type="user"))
        return await _active_item_count(repo, user), await _job_status(repo, user)

    items, status = asyncio.run(_run())
    assert items == 0  # 不变量：最终 active items 必须为 0
    assert status == "succeeded"  # 合法终态之一


def test_delete_all_before_apply_cancels_job(repo):
    """线性化顺序 2：delete-all 先提交 → worker apply 读到新 generation → cancelled。

    模拟真实场景：worker 已领取（processing，delete-all 不取消 processing job），
    delete-all 令 generation+1，worker 处理时 generation 不匹配 → cancelled。
    """
    user = "apply-2"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        # worker 领取 → processing（delete-all 不取消 processing job）
        claimed = await repo.claim_jobs(
            "w1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        job_id = claimed[0].job_id
        # delete-all：generation 0 → 1
        await repo.delete_all_memories(user, ActorContext(actor_type="user"))
        # worker 用旧 generation=0 预期 apply → cancelled
        outcome = await repo.apply_extract_job(
            job_id=job_id,
            worker_id="w1",
            user_id=user,
            expected_generation=0,  # 旧 generation（delete-all 前入队）
            commands=[_tool_verified_command(repo, user)],
            actor=ActorContext(actor_type="system", source_thread_id="t1"),
        )
        return outcome, await _active_item_count(repo, user), await _job_status(repo, user)

    outcome, items, status = asyncio.run(_run())
    assert outcome == "cancelled"
    assert items == 0
    assert status == "cancelled"


def test_stale_worker_rejected(repo):
    """#1.1：lease 被 reaper 回收、另一 worker 接手后，旧 worker apply 被拒（stale）。"""
    user = "apply-stale"

    async def _run():
        job_id = await repo.enqueue_job(_extract_command(user))
        # worker A claim
        claimed_a = await repo.claim_jobs(
            "wA", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        # A 卡住：lease 超时 → reaper 回收 → pending
        await repo.recover_due_jobs(
            datetime.now(timezone.utc) + timedelta(seconds=601),
            lease_seconds=300, max_attempts=5,
        )
        # worker B 重新 claim
        claimed_b = await repo.claim_jobs(
            "wB", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        assert any(j.job_id == job_id for j in claimed_b)
        # 旧 worker A 恢复并尝试 apply → stale（不写记忆）
        outcome = await repo.apply_extract_job(
            job_id=job_id,
            worker_id="wA",  # 锁已被 B 持有
            user_id=user,
            expected_generation=0,
            commands=[_tool_verified_command(repo, user)],
            actor=ActorContext(actor_type="system", source_thread_id="t1"),
        )
        return outcome, await _active_item_count(repo, user)

    outcome, items = asyncio.run(_run())
    assert outcome == "stale"
    assert items == 0  # 旧 worker 不得写入


# ============================================================
# #3: 本轮区间提取
# ============================================================

def _round_worker(repo, user: str, messages: list[dict]):
    return MemoryWorker(
        repo,
        worker_id="w1",
        memory_service=_service(repo),
        display_messages_loader=lambda thread_id: messages,
    )


def test_worker_only_extracts_current_round(repo):
    """#3：thread 有旧 order_create，但本轮无工具事件 → 不刷新旧记忆。"""
    user = "round-1"

    async def _run():
        # 上一轮：order_create（会产生 tool_verified 记忆）
        await _service(repo).remember(
            user, ActorContext(actor_type="system", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="订单创建成功：PO-1",
            data={"task_type": "订单创建", "result_status": "succeeded"},
            source_type=SourceType.TOOL_VERIFIED,
        )
        # 当前轮：user + assistant，无工具事件（旧 tool 消息在区间外）
        await repo.enqueue_job(_extract_command(user))
        messages = [
            {"id": "user-old", "role": "user", "content": "上轮"},
            {"id": "tool-old", "role": "tool", "tool_name": "order_create", "content": "PO-1 创建成功"},
            {"id": "assistant-old", "role": "assistant", "content": "上轮完成"},
            {"id": "user-1", "role": "user", "content": "这轮只是闲聊"},
            {"id": "assistant-1", "role": "assistant", "content": "好的"},
        ]
        worker = _round_worker(repo, user, messages)
        await worker.run_once()
        # 旧记忆只应存在一条（未被重复刷新）
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        # 检查旧记忆 updated_at 未变：直接数 items
        return len(page.items), await _job_status(repo, user)

    items, status = asyncio.run(_run())
    assert status == "succeeded"
    assert items == 1  # 只有上轮的旧记忆，本轮无新写入


def test_worker_missing_referenced_message_fails(repo):
    """#3：引用消息缺失 → failed（重试），不 succeeded。"""
    user = "round-missing"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        worker = _round_worker(repo, user, [
            {"id": "other-1", "role": "user", "content": "别的消息"},
        ])
        await worker.run_once()
        return await _job_status(repo, user)

    status = asyncio.run(_run())
    assert status == "failed"  # 不伪造成功


# ============================================================
# #4: processing lease 回收
# ============================================================

def test_processing_lease_expired_recovered(repo):
    """#4：processing 超时 → reaper 恢复 pending → 其他 worker 可重新 claim。"""
    user = "lease-1"

    async def _run():
        job_id = await repo.enqueue_job(_extract_command(user))
        # worker A claim → processing
        await repo.claim_jobs(
            "wA", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        # A crash：lease（300s）超时后 reaper 恢复
        await repo.recover_due_jobs(
            datetime.now(timezone.utc) + timedelta(seconds=601),
            lease_seconds=300, max_attempts=5,
        )
        # worker B 可重新 claim
        claimed = await repo.claim_jobs(
            "wB", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        return any(j.job_id == job_id for j in claimed), len(claimed)

    reclaimable, count = asyncio.run(_run())
    assert reclaimable is True
    assert count == 1


def test_processing_lease_dead_after_max_attempts(repo):
    """#4：processing 超时且 attempts 耗尽 → dead（终态，不可再 claim）。"""
    user = "lease-dead"

    async def _run():
        job_id = await repo.enqueue_job(_extract_command(user))
        # 领取并失败 5 次（attempts 达到 max）
        for i in range(5):
            claimed = await repo.claim_jobs(
                "wA", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
            )
            if not claimed:
                break
            job = claimed[0]
            await repo.fail_job(
                job.job_id, "wA", "失败",
                datetime.now(timezone.utc) - timedelta(seconds=1),
            )
            await repo.recover_due_jobs(
                datetime.now(timezone.utc), lease_seconds=300, max_attempts=5,
            )
        # attempts 耗尽后超时 → dead
        await repo.recover_due_jobs(
            datetime.now(timezone.utc) + timedelta(seconds=601),
            lease_seconds=300, max_attempts=5,
        )
        status = await _job_status(repo, user)
        return status

    status = asyncio.run(_run())
    assert status == "dead"


def _key(user: str) -> str:
    from agent.memory.repository import compute_idempotency_key
    return compute_idempotency_key(
        job_type="extract_memory", user_id=user, thread_id="t1",
        assistant_message_id="assistant-1", extractor_version="memory-v2.1",
        memory_generation=0, replay_generation=0,
    )


# ============================================================
# review 第二轮 #2: apply_expire_job generation fence
# ============================================================

def _expire_command(user: str, cutoff: str) -> EnqueueJobCommand:
    return EnqueueJobCommand(
        user_id=user,
        thread_id="t1",
        job_type="expire_memory",
        payload={
            "scan_partition": "p0",
            "cutoff_at": cutoff,
            "policy_version": "memory-v2.1",
            "memory_generation": 0,
            "replay_generation": 0,
        },
    )


async def _expired_seed(repo, user: str) -> None:
    """写入一条已过期（expires_at 早于 cutoff）的 item。

    CHECK 约束 expires_at >= created_at（17.1）阻止直接写过去的 expires_at，
    用 SQL 同时前移 created_at（10 天前创建、1 天前过期）满足约束。
    """
    from agent.memory.repository import CreateMemoryCommand
    from agent.memory.models import compute_fingerprint, new_uuid7
    cmd = CreateMemoryCommand(
        user_id=user,
        memory_id=new_uuid7(),
        kind=MemoryKind.TASK_OUTCOME,
        content="过期任务结果",
        data={"task_type": "询价", "result_status": "succeeded"},
        source_type=SourceType.TOOL_VERIFIED,
        source_thread_id="t1",
        fingerprint=compute_fingerprint(
            MemoryKind.TASK_OUTCOME, None, None, "过期任务结果",
            {"task_type": "询价", "result_status": "succeeded"},
        ),
    )
    await repo.create_or_resolve_memory(cmd, ActorContext(actor_type="system", source_thread_id="t1"))
    # 模拟时间流逝：10 天前创建、1 天前已过期（created_at 与 expires_at 同时前移满足 CHECK）
    engine = create_async_engine(DSN)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE memory_items SET "
                "created_at = created_at - INTERVAL 10 DAY, "
                "expires_at = created_at + INTERVAL 9 DAY "
                "WHERE user_id = :u"
            ),
            {"u": user},
        )
    await engine.dispose()


def test_apply_expire_job_cleans_expired(repo):
    """expire job：过期 item 被 forgotten，revision 递增，job succeeded。"""
    user = "expire-ok"

    async def _run():
        await _expired_seed(repo, user)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        await repo.enqueue_job(_expire_command(user, cutoff))
        claimed = await repo.claim_jobs(
            "w1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        outcome = await repo.apply_expire_job(
            job_id=claimed[0].job_id, worker_id="w1", user_id=user,
            expected_generation=0, cutoff=cutoff.isoformat(),
        )
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        return outcome, len(page.items), await _job_status(repo, user)

    outcome, items, status = asyncio.run(_run())
    assert outcome == "succeeded"
    assert items == 0  # 过期 item 已停止召回
    assert status == "succeeded"


def test_apply_expire_job_generation_mismatch_cancelled(repo):
    """delete-all 后旧 generation 的 expire job → cancelled，不修改新状态。"""
    user = "expire-gen"

    async def _run():
        await _expired_seed(repo, user)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        await repo.enqueue_job(_expire_command(user, cutoff))
        claimed = await repo.claim_jobs(
            "w1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        # delete-all：generation 0 → 1
        await repo.delete_all_memories(user, ActorContext(actor_type="user"))
        outcome = await repo.apply_expire_job(
            job_id=claimed[0].job_id, worker_id="w1", user_id=user,
            expected_generation=0, cutoff=cutoff.isoformat(),  # 旧 generation
        )
        return outcome, await _job_status(repo, user)

    outcome, status = asyncio.run(_run())
    assert outcome == "cancelled"
    assert status == "cancelled"


def test_apply_expire_job_stale_worker(repo):
    """lease 回收后旧 worker expire → stale，不修改记忆。"""
    user = "expire-stale"

    async def _run():
        await _expired_seed(repo, user)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        await repo.enqueue_job(_expire_command(user, cutoff))
        # worker A claim → A 卡住 → lease 超时 → reaper 回收 → B 重新 claim
        await repo.claim_jobs(
            "wA", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        await repo.recover_due_jobs(
            datetime.now(timezone.utc) + timedelta(seconds=601),
            lease_seconds=300, max_attempts=5,
        )
        # worker B 重新 claim（持有锁）
        claimed_b = await repo.claim_jobs(
            "wB", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        assert len(claimed_b) == 1
        # 旧 worker A 恢复 → stale（锁已被 B 持有）
        outcome = await repo.apply_expire_job(
            job_id=claimed_b[0].job_id,
            worker_id="wA",
            user_id=user, expected_generation=0, cutoff=cutoff.isoformat(),
        )
        # 过期 item 未被清理（stale 不写）
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        return outcome, len(page.items)

    outcome, items = asyncio.run(_run())
    assert outcome == "stale"
    assert items == 1  # 过期 item 仍在（未被清理）
