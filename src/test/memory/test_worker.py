# src/test/memory/test_worker.py
"""MemoryWorker 集成测试（方案 19.3）。

覆盖：generation 匹配处理成功、generation 不匹配 cancelled（不写记忆）、
失败退避重试、超过最大尝试转 dead、worker 模式校验。
使用独立测试库 memory_v2_worker_test。
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

from agent.memory.repository import (  # noqa: E402
    EnqueueJobCommand,
    MySQLMemoryRepository,
)
from agent.memory.worker import MemoryWorker  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_worker_test"
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
    """每个测试后清空 jobs 表。"""
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


def _extract_command(user: str, generation: int = 0) -> EnqueueJobCommand:
    return EnqueueJobCommand(
        user_id=user,
        thread_id="t1",
        job_type="extract_memory",
        payload={
            "checkpoint_id": "cp-1",
            "user_message_id": "um-1",
            "assistant_message_id": "am-1",
            "extractor_version": "memory-v2.1",
            "memory_generation": generation,
            "replay_generation": 0,
        },
    )


async def _job_status(repo, user: str) -> str:
    jobs = await repo.claim_jobs("probe", limit=100, now=datetime.now(timezone.utc) + __import__("datetime").timedelta(seconds=1))
    # 直接查表
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


def test_worker_missing_dependencies_fails_job(repo):
    """review #2：缺少 display_messages_loader/memory_service 时 Job 必须 failed
    （重试），不能伪造 succeeded。"""
    user = "wk-nodep-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user, generation=0))
        worker = MemoryWorker(repo, worker_id="w1")
        processed = await worker.run_once()
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT status FROM memory_jobs WHERE user_id=:u"),
                    {"u": user},
                )
            ).first()
        await engine.dispose()
        return processed, row[0]

    processed, status = asyncio.run(_run())
    assert processed == 1
    assert status == "failed"  # 不是 succeeded


def test_worker_cancels_job_on_generation_mismatch(repo):
    """generation 不匹配（delete-all 后 processing 旧 job）→ cancelled，不写记忆。

    真实场景：worker 已领取（processing），delete-all 令 generation+1，
    worker 处理时发现不匹配 → cancelled（方案 20.4：processing Job 由 worker 检查）。
    """
    user = "wk-gen-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user, generation=0))
        # worker-1 领取 → processing（delete-all 不取消 processing job）
        claimed = await repo.claim_jobs(
            "w1", limit=20, now=datetime.now(timezone.utc) + timedelta(seconds=1)
        )
        assert len(claimed) == 1
        from agent.memory.repository import ActorContext
        await repo.delete_all_memories(user, ActorContext(actor_type="user"))
        # worker 处理时 generation 检查失败 → 走 cancelled 路径
        worker = MemoryWorker(repo, worker_id="w1")
        try:
            await worker._process_job(claimed[0])
        except Exception as exc:
            from agent.memory.worker import JobGenerationMismatch
            assert isinstance(exc, JobGenerationMismatch)
            await worker._cancel_job(claimed[0])
        status = await _job_status(repo, user)
        # 确认没有复活记忆
        from agent.memory.repository import MemoryListFilter
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        return status, len(page.items)

    status, item_count = asyncio.run(_run())
    assert status == "cancelled"
    assert item_count == 0  # 未写 Profile/Item/Event


def test_worker_failure_retries_with_backoff(repo):
    """处理失败 → failed + 未来重试时间。"""
    user = "wk-fail-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))

        class FailingWorker(MemoryWorker):
            async def _handle_extract(self, job):
                raise RuntimeError("模拟抽取失败")

        worker = FailingWorker(repo, worker_id="w1")
        await worker.run_once()
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text(
                        "SELECT status, available_at, attempts, last_error "
                        "FROM memory_jobs WHERE user_id=:u"
                    ),
                    {"u": user},
                )
            ).first()
        await engine.dispose()
        return row

    row = asyncio.run(_run())
    assert row.status == "failed"
    assert row.attempts == 1
    assert row.available_at > datetime.now(timezone.utc).replace(tzinfo=None)
    # 安全摘要（方案 20.3）：只含受控 reason code + 类型名，不含 raw exception
    assert "reason_code=unexpected_error" in row.last_error
    assert "error_type=RuntimeError" in row.last_error
    assert "模拟抽取失败" not in row.last_error


def test_worker_dead_after_max_attempts(repo):
    """超过最大尝试 → dead（终态）。"""
    user = "wk-dead-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))

        class FailingWorker(MemoryWorker):
            async def _handle_extract(self, job):
                raise RuntimeError("总是失败")

        worker = FailingWorker(repo, worker_id="w1", max_attempts=2)
        # 第一次失败（attempts=1，退避中）
        await worker.run_once()
        # 退避到期后重试（attempts=2 >= max → dead；用未来时间模拟到期）
        future = datetime.now(timezone.utc) + timedelta(seconds=600)
        await worker.run_once(now=future)
        status = await _job_status(repo, user)
        return status

    status = asyncio.run(_run())
    assert status == "dead"


def test_worker_last_error_never_contains_sensitive_body(repo):
    """方案 20.3：job last_error 不保存 raw exception（SQL 参数/记忆正文/凭据）。"""
    user = "wk-sec-1"

    async def _run():
        await repo.enqueue_job(_extract_command(user))

        class LeakyWorker(MemoryWorker):
            async def _handle_extract(self, job):
                raise RuntimeError(
                    "connection failed SELECT ... password=TOP_SECRET_123"
                )

        worker = LeakyWorker(repo, worker_id="w1", max_attempts=2)
        await worker.run_once()
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            row = (
                await conn.execute(
                    text("SELECT status, last_error FROM memory_jobs WHERE user_id=:u"),
                    {"u": user},
                )
            ).first()
        await engine.dispose()
        return row

    row = asyncio.run(_run())
    assert row.status == "failed"
    assert "TOP_SECRET_123" not in row.last_error
    assert "password=" not in row.last_error
    # 安全摘要包含受控 reason code + 类型名
    assert "reason_code=unexpected_error" in row.last_error
    assert "error_type=RuntimeError" in row.last_error


def test_safe_error_summary_redacts_sensitive_body():
    """方案 20.3：日志/错误摘要来自 safe_error_summary（纯函数），不泄漏正文。"""
    from agent.memory.worker import safe_error_summary

    # RuntimeError 携带 SQL 与凭据 → 摘要只含受控 reason code + 类型名
    summary = safe_error_summary(RuntimeError("SQL with password=TOP_SECRET_789"))
    assert "TOP_SECRET_789" not in summary
    assert "password=" not in summary
    assert "reason_code=unexpected_error" in summary
    assert "error_type=RuntimeError" in summary

    # Pydantic ValidationError（含 input_value）→ validation_failed
    from agent.memory.models import MemoryCandidate
    try:
        MemoryCandidate.model_validate(
            {"kind": "bogus", "content": "x", "extraction_confidence": 0.9}
        )
        raise AssertionError("应校验失败")
    except Exception as exc:
        summary = safe_error_summary(exc)
        assert "bogus" not in summary
        assert "reason_code=validation_failed" in summary



def test_worker_mode_validation():
    """MEMORY_WORKER_MODE 非法值拒绝。"""
    import importlib
    import agent.memory.worker as worker_module

    old = worker_module.WORKER_MODE
    try:
        worker_module.WORKER_MODE = "bogus"
        with pytest.raises(RuntimeError):
            worker_module.worker_mode()
        worker_module.WORKER_MODE = "embedded"
        assert worker_module.worker_mode() == "embedded"
        worker_module.WORKER_MODE = "standalone"
        assert worker_module.worker_mode() == "standalone"
        worker_module.WORKER_MODE = "disabled"
        assert worker_module.worker_mode() == "disabled"
    finally:
        worker_module.WORKER_MODE = old
