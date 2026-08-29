# src/test/memory/test_extractor.py
"""Structured extractor 测试（方案 4.3 / 6.3 / 6.4 / 18.3）。

覆盖：抽取阈值过滤（>=0.85 入库、[0.65,0.85) 与 <0.65 丢弃）、
确定性工具事件 → tool_verified 候选、worker 自动路径只入库 tool_verified
（model_inferred 第一版一律丢弃）。
使用独立测试库 memory_v2_extractor_test。
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

from agent.memory.extractor import (  # noqa: E402
    StructuredExtractor,
    apply_extraction_threshold,
    tool_event_candidates,
)
from agent.memory.models import (  # noqa: E402
    MemoryCandidate,
    MemoryExtractionResult,
    MemoryKind,
    ProfilePatch,
)
from agent.memory.policies import MemoryPolicy  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    EnqueueJobCommand,
    MemoryListFilter,
    MySQLMemoryRepository,
)
from agent.memory.service import MemoryService  # noqa: E402
from agent.memory.worker import MemoryWorker  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_extractor_test"
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


# ============================================================
# 抽取阈值（方案 18.3）
# ============================================================

def _candidate(confidence: float) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.TASK_OUTCOME,
        content=f"候选 {confidence}",
        data={"task_type": "询价", "result_status": "succeeded"},
        extraction_confidence=confidence,
    )


def test_threshold_keeps_high_confidence_only():
    result = MemoryExtractionResult(
        memory_candidates=[
            _candidate(0.9),   # >= 0.85 保留
            _candidate(0.85),  # 边界保留
            _candidate(0.8),   # [0.65, 0.85) 丢弃
            _candidate(0.65),  # 边界丢弃
            _candidate(0.5),   # < 0.65 丢弃
        ],
    )
    filtered = apply_extraction_threshold(result)
    assert len(filtered.memory_candidates) == 2
    assert all(c.extraction_confidence >= 0.85 for c in filtered.memory_candidates)


# ============================================================
# 确定性工具事件（方案 6.3 第 1 条）
# ============================================================

def test_tool_event_candidates_from_order_success():
    """review 第二轮：succeeded 显式为 True 才生成候选。"""
    candidates = tool_event_candidates(
        [{"tool_name": "order_create", "summary": "订单 PO-1 创建成功", "succeeded": True}]
    )
    assert len(candidates) == 1
    c = candidates[0]
    assert c.kind == MemoryKind.TASK_OUTCOME
    assert c.extraction_confidence == 1.0  # 确定性事件
    assert "订单创建成功" in c.content


def test_tool_event_failed_not_generated():
    """succeeded=False（异常失败）→ 不生成确定性候选。"""
    assert tool_event_candidates(
        [{"tool_name": "order_create", "summary": "订单创建失败", "succeeded": False}]
    ) == []


def test_tool_event_unconfirmed_not_generated():
    """无 succeeded 字段（业务失败无法从文本确认）→ 不生成（禁止文本猜测）。"""
    assert tool_event_candidates(
        [{"tool_name": "order_create", "summary": "库存不足，订单创建失败"}]
    ) == []


def test_tool_event_unknown_tool_ignored():
    assert tool_event_candidates(
        [{"tool_name": "web_search", "summary": "...", "succeeded": True}]
    ) == []


# ============================================================
# StructuredExtractor（fake model）
# ============================================================

class FakeModel:
    """模拟 with_structured_output 的模型。"""

    def __init__(self, payload: dict):
        self._payload = payload

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        return self._payload


def test_extractor_parses_model_output():
    extractor = StructuredExtractor(
        FakeModel(
            {
                "profile_patches": [],
                "memory_candidates": [
                    {
                        "kind": "procurement_constraint",
                        "content": "用户要求交期不超过 14 天",
                        "data": {"constraint_name": "delivery_days_max", "value": 14},
                        "extraction_confidence": 0.9,
                    }
                ],
            }
        )
    )

    async def _run():
        return await extractor.extract(
            user_message="以后交期不要超过 14 天",
            assistant_message="好的",
        )

    result = asyncio.run(_run())
    assert len(result.memory_candidates) == 1
    assert result.memory_candidates[0].content == "用户要求交期不超过 14 天"


def test_extractor_rejects_invalid_model_output():
    """模型返回错误结构（未知字段/非法 kind）→ ValidationError（整批拒绝，6.3 第 4 条）。"""
    extractor = StructuredExtractor(
        FakeModel(
            {
                "profile_patches": [],
                "memory_candidates": [
                    {
                        "kind": "bogus_kind",  # 非法枚举
                        "content": "x",
                        "extraction_confidence": 0.9,
                    }
                ],
            }
        )
    )

    async def _run():
        return await extractor.extract(user_message="x", assistant_message="y")

    with pytest.raises(Exception):
        asyncio.run(_run())


# ============================================================
# worker 自动路径（方案 6.4：只入库 tool_verified）
# ============================================================

def test_worker_auto_path_persists_tool_verified_only(repo):
    """worker 抽取：工具事件候选入库；模型候选（model_inferred）丢弃。"""
    user = "ext-wk-1"

    async def _run():
        await repo.enqueue_job(
            EnqueueJobCommand(
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
        )
        # 展示消息：本轮区间 user → tool → assistant（工具调用在回复前）。
        # 生产形态（review 第三轮 producer/consumer 契约）：结果存 text 字段、
        # succeeded 由 chat.py 在 ToolMessage 处生成
        display_messages = [
            {"id": "user-1", "role": "user", "content": "创建订单"},
            {"id": "tool-1", "role": "tool", "tool_name": "order_create",
             "text": "PO-1 创建成功", "succeeded": True},
            {"id": "assistant-1", "role": "assistant", "content": "订单已创建"},
        ]
        memory_service = MemoryService(repo, MemoryPolicy())
        worker = MemoryWorker(
            repo,
            worker_id="w1",
            memory_service=memory_service,
            display_messages_loader=lambda thread_id: display_messages,
        )
        await worker.run_once()
        # 诊断：job 状态
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            job_row = (
                await conn.execute(
                    text("SELECT status, last_error FROM memory_jobs WHERE user_id=:u"),
                    {"u": user},
                )
            ).first()
        await engine.dispose()
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        return job_row, page

    job_row, page = asyncio.run(_run())
    assert job_row.status == "succeeded", f"job 未成功: {job_row}"
    assert len(page.items) == 1
    item = page.items[0]
    assert item.source_type.value == "tool_verified"
    assert item.kind == MemoryKind.TASK_OUTCOME
    assert "订单创建成功" in item.content


# ============================================================
# review 第三轮：producer 端 succeeded 生成契约（chat.py）
# ============================================================

def test_tool_result_error_envelope_detected():
    """ERP 工具返回契约：{"error": ...} → has_error=True（succeeded=False）。"""
    from api_view.api.chat import _tool_result_has_error

    # JSON 文本形态（MCP 层序列化）
    assert _tool_result_has_error('{"error": "API error: code=500, message=库存不足"}') is True
    # dict 形态
    assert _tool_result_has_error({"error": "创建失败"}) is True
    # content blocks 形态（langchain_mcp_adapters）
    assert _tool_result_has_error([{"type": "text", "text": '{"error": "参数错误"}'}]) is True


def test_tool_result_success_detected():
    """ERP 工具返回契约：data dict（无 error 键）→ has_error=False（succeeded=True）。"""
    from api_view.api.chat import _tool_result_has_error

    assert _tool_result_has_error('{"data": {"orderId": 1001}}') is False
    assert _tool_result_has_error({"data": {"orderId": 1001}}) is False


def test_tool_result_unstructured_returns_none():
    """纯文本（无法结构化判定）→ None（不写 succeeded，不生成候选）。"""
    from api_view.api.chat import _tool_result_has_error

    assert _tool_result_has_error("PO-1 创建成功") is None
    assert _tool_result_has_error(123) is None
    assert _tool_result_has_error(None) is None


def test_worker_production_contract_success(repo):
    """生产形态全链路：text 字段 + succeeded=True → 1 条 tool_verified，summary 非空。"""
    user = "prod-contract-1"

    async def _run():
        await repo.enqueue_job(
            EnqueueJobCommand(
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
        )
        display_messages = [
            {"id": "user-1", "role": "user", "content": "创建订单"},
            {"id": "tool-1", "role": "tool", "tool_name": "order_create",
             "text": '{"data": {"orderId": 1001, "orderNumber": "PO20260808001"}}',
             "succeeded": True},
            {"id": "assistant-1", "role": "assistant", "content": "订单已创建"},
        ]
        memory_service = MemoryService(repo, MemoryPolicy())
        worker = MemoryWorker(
            repo,
            worker_id="w1",
            memory_service=memory_service,
            display_messages_loader=lambda thread_id: display_messages,
        )
        await worker.run_once()
        page = await repo.list_memories(user, MemoryListFilter(), None, 50)
        return page

    page = asyncio.run(_run())
    assert len(page.items) == 1
    item = page.items[0]
    assert item.source_type.value == "tool_verified"
    # summary 来自 text 字段（非空，携带订单信息，不同订单不产生相同指纹）
    assert "PO20260808001" in item.content
