# src/test/memory/test_jobs.py
"""Job queue 集成测试（方案 19.1 / 19.2 / 19.3）。

覆盖：入队（generation 写入 payload）、幂等键（同 key 幂等、checkpoint_id 不参与）、
payload 类型化校验、SKIP LOCKED 领取（不重复）、complete/fail 状态机。
使用独立测试库 memory_v2_jobs_test。
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

from agent.memory.models import validate_job_payload  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    EnqueueJobCommand,
    MySQLMemoryRepository,
    compute_idempotency_key,
)

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_jobs_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

NOW = datetime.now(timezone.utc)


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
    """每个测试后清空 jobs 表（claim_jobs 是全表领取，避免跨测试累积）。"""
    yield
    async def _clean():
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            await conn.execute(text("DELETE FROM memory_jobs"))
        await engine.dispose()
    asyncio.run(_clean())


@pytest.fixture()
def repo():
    engine = create_async_engine(DSN)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield MySQLMemoryRepository(factory)
    asyncio.run(engine.dispose())


def _extract_command(user: str, *, assistant_message_id: str = "am-1") -> EnqueueJobCommand:
    return EnqueueJobCommand(
        user_id=user,
        thread_id="t1",
        job_type="extract_memory",
        payload={
            "checkpoint_id": "cp-1",
            "user_message_id": "um-1",
            "assistant_message_id": assistant_message_id,
            "extractor_version": "memory-v2.1",
            "memory_generation": 0,
            "replay_generation": 0,
        },
    )


# ============================================================
# 入队与幂等（方案 19.1 / 19.2）
# ============================================================

def test_enqueue_job_writes_generation(repo):
    """入队时从 memory_user_state 读取 generation 写入 payload。"""
    user = "jobs-gen-1"

    async def _run():
        job_id = await repo.enqueue_job(_extract_command(user))
        job = await repo.get_job_by_idempotency_key(
            compute_idempotency_key(
                job_type="extract_memory", user_id=user, thread_id="t1",
                assistant_message_id="am-1", extractor_version="memory-v2.1",
                memory_generation=0, replay_generation=0,
            )
        )
        return job_id, job

    job_id, job = asyncio.run(_run())
    assert job_id == job.job_id
    assert job.status == "pending"
    assert job.payload["memory_generation"] == 0


def test_enqueue_idempotent_same_key(repo):
    """同一幂等键重复入队 → 返回同一 job_id，不重复插入（19.2）。"""
    user = "jobs-idem-1"

    async def _run():
        job_id_1 = await repo.enqueue_job(_extract_command(user))
        job_id_2 = await repo.enqueue_job(_extract_command(user))
        return job_id_1, job_id_2

    j1, j2 = asyncio.run(_run())
    assert j1 == j2


def test_idempotency_key_ignores_checkpoint_id(repo):
    """checkpoint_id 不参与 extract_memory 幂等键（19.2：恢复/重试换 checkpoint 不重复建任务）。"""
    user = "jobs-cp-1"

    async def _run():
        cmd1 = _extract_command(user)
        cmd2 = _extract_command(user)
        cmd2.payload["checkpoint_id"] = "cp-CHANGED"
        k1 = compute_idempotency_key(
            job_type="extract_memory", user_id=user, thread_id="t1",
            assistant_message_id="am-1", extractor_version="memory-v2.1",
            memory_generation=0, replay_generation=0,
        )
        return await repo.enqueue_job(cmd1), await repo.enqueue_job(cmd2), k1

    j1, j2, k1 = asyncio.run(_run())
    assert j1 == j2  # checkpoint 变化不影响幂等键


def test_idempotency_key_changes_with_user_or_message(repo):
    """改变 user_id / assistant_message_id → 不同 key（19.2 跨用户/任务无碰撞）。"""
    k1 = compute_idempotency_key(
        job_type="extract_memory", user_id="u1", thread_id="t1",
        assistant_message_id="am-1", extractor_version="memory-v2.1",
        memory_generation=0, replay_generation=0,
    )
    k2 = compute_idempotency_key(
        job_type="extract_memory", user_id="u2", thread_id="t1",
        assistant_message_id="am-1", extractor_version="memory-v2.1",
        memory_generation=0, replay_generation=0,
    )
    k3 = compute_idempotency_key(
        job_type="extract_memory", user_id="u1", thread_id="t1",
        assistant_message_id="am-2", extractor_version="memory-v2.1",
        memory_generation=0, replay_generation=0,
    )
    assert k1 != k2 and k1 != k3


# ============================================================
# payload 类型化校验（方案 19.1：extra=forbid）
# ============================================================

def test_extract_payload_requires_message_ids():
    """extract_memory payload 必填消息 ID；缺失拒绝。"""
    with pytest.raises(Exception):
        validate_job_payload(
            "extract_memory",
            {"checkpoint_id": "cp", "extractor_version": "v1"},
        )


def test_expire_payload_rejects_message_fields():
    """expire_memory payload 拒绝消息字段（19.1：不得包含消息 ID）。"""
    with pytest.raises(Exception):
        validate_job_payload(
            "expire_memory",
            {
                "scan_partition": "p0",
                "cutoff_at": NOW,
                "policy_version": "v1",
                "memory_generation": 0,
                "replay_generation": 0,
                "assistant_message_id": "am-1",  # 非法字段
            },
        )


def test_expire_payload_valid():
    payload = validate_job_payload(
        "expire_memory",
        {
            "scan_partition": "p0",
            "cutoff_at": NOW,
            "policy_version": "v1",
            "memory_generation": 0,
            "replay_generation": 0,
        },
    )
    assert payload["scan_partition"] == "p0"


# ============================================================
# 领取与状态机（方案 19.3）
# ============================================================

def test_claim_jobs_sets_lock_and_attempts(repo):
    user = "jobs-claim-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        claimed = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        return claimed

    claimed = asyncio.run(_run())
    assert len(claimed) == 1
    job = claimed[0]
    assert job.status == "processing"
    assert job.locked_by == "worker-1"
    assert job.attempts == 1


def test_claim_skips_locked_jobs(repo):
    """SKIP LOCKED：worker-1 领取后，worker-2 不再领取同一 job。"""
    user = "jobs-skip-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        first = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        second = await repo.claim_jobs("worker-2", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        return first, second

    first, second = asyncio.run(_run())
    assert len(first) == 1
    assert len(second) == 0  # processing 中不可再领取


def test_complete_job_clears_lock(repo):
    user = "jobs-done-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        claimed = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        await repo.complete_job(claimed[0].job_id, "worker-1")
        # 通过幂等键查回
        job = await repo.get_job_by_idempotency_key(
            compute_idempotency_key(
                job_type="extract_memory", user_id=user, thread_id="t1",
                assistant_message_id="am-1", extractor_version="memory-v2.1",
                memory_generation=0, replay_generation=0,
            )
        )
        return job

    job = asyncio.run(_run())
    assert job.status == "succeeded"
    assert job.locked_by is None


def test_fail_job_sets_retry_at_and_error(repo):
    user = "jobs-fail-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        claimed = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=30)
        await repo.fail_job(claimed[0].job_id, "worker-1", "抽取超时", retry_at)
        job = await repo.get_job_by_idempotency_key(
            compute_idempotency_key(
                job_type="extract_memory", user_id=user, thread_id="t1",
                assistant_message_id="am-1", extractor_version="memory-v2.1",
                memory_generation=0, replay_generation=0,
            )
        )
        return job, retry_at

    job, retry_at = asyncio.run(_run())
    assert job.status == "failed"
    assert job.locked_by is None
    assert job.available_at == retry_at
    assert job.last_error == "抽取超时"


def test_failed_job_requeued_after_retry_at(repo):
    """failed 且 available_at 到期后可再次被领取（19.3 重试）。"""
    user = "jobs-retry-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))
        claimed1 = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        await repo.fail_job(
            claimed1[0].job_id, "worker-1", "失败",
            datetime.now(timezone.utc) - timedelta(seconds=1),  # 已到期
        )
        # reaper：到期 failed → pending（方案 19.3 重试流转）
        await repo.recover_due_jobs(datetime.now(timezone.utc), lease_seconds=300, max_attempts=5)
        claimed2 = await repo.claim_jobs("worker-1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1))
        return claimed2

    claimed2 = asyncio.run(_run())
    assert len(claimed2) == 1
    assert claimed2[0].attempts == 2  # 重试 attempts 递增
