# src/test/test_char_checkpoint.py
"""MongoDB checkpoint 链路 characterization tests。

Memory v2 升级期间必须保留的行为（方案 21.3 第 1 步）：
- MongoDBSaver 保存/恢复 thread checkpoint
- 跨"进程重启"（新 Saver 实例）后同一 thread 可恢复
- 不同 thread 之间隔离
- HITL 中断状态（pending_sends）随 checkpoint 持久化

使用真实 MongoDB 上的一次性 collection（checkpoints_char_tmp），
teardown 时 drop，不触碰生产 collection。

与仓库现有测试一致：同步测试函数 + 内部 asyncio.run（无 pytest-asyncio）。
"""
import asyncio
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.checkpoint.base import (
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from agent.config import MONGODB_DB_NAME, MONGODB_URI

TEST_COLLECTION = "checkpoints_char_tmp"
TEST_WRITES_COLLECTION = "checkpoint_writes_char_tmp"


@pytest.fixture()
def client():
    return MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}


def _make_checkpoint(channel_values: dict, pending_sends: list | None = None) -> dict:
    """构造与 LangGraph 运行时写入形态一致的 checkpoint dict。"""
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = channel_values
    checkpoint["versions_seen"] = {"__start__": {"__start__": 1}}
    if pending_sends is not None:
        checkpoint["pending_sends"] = pending_sends
    return checkpoint


def _make_metadata() -> CheckpointMetadata:
    return CheckpointMetadata(source="input", step=1, writes=None, parents={})


async def _roundtrip(client, saver) -> None:
    thread_id = f"char-roundtrip-{uuid.uuid4()}"
    interrupt_payload = [("call-abc", {"type": "order_info_request", "missing_fields": ["a"]})]
    checkpoint = _make_checkpoint(
        channel_values={"messages": ["hello"], "count": 3},
        pending_sends=interrupt_payload,
    )

    saved = await saver.aput(
        _thread_config(thread_id), checkpoint, _make_metadata(), new_versions={}
    )

    loaded = await saver.aget_tuple(_thread_config(thread_id))
    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == {"messages": ["hello"], "count": 3}
    # MongoDB 序列化后 pending_sends 的 tuple 变 list，值与结构等价（运行时下游行为一致）
    assert [tuple(item) for item in loaded.checkpoint["pending_sends"]] == interrupt_payload
    assert saved["configurable"]["checkpoint_id"] == loaded.checkpoint["id"]


async def _restart_recovery(client) -> None:
    thread_id = f"char-restart-{uuid.uuid4()}"
    checkpoint = _make_checkpoint(channel_values={"messages": ["old"]})
    try:
        saver1 = MongoDBSaver(
            client=client,
            db_name=MONGODB_DB_NAME,
            checkpoint_collection_name=TEST_COLLECTION,
            writes_collection_name=TEST_WRITES_COLLECTION,
        )
        await saver1.aput(_thread_config(thread_id), checkpoint, _make_metadata(), new_versions={})

        # 模拟重启：全新 Saver 实例 + 全新 MongoClient
        client2 = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        saver2 = MongoDBSaver(
            client=client2,
            db_name=MONGODB_DB_NAME,
            checkpoint_collection_name=TEST_COLLECTION,
            writes_collection_name=TEST_WRITES_COLLECTION,
        )
        loaded = await saver2.aget_tuple(_thread_config(thread_id))
        assert loaded is not None
        assert loaded.checkpoint["channel_values"]["messages"] == ["old"]
        client2.close()
    finally:
        client[MONGODB_DB_NAME][TEST_COLLECTION].drop()
        client[MONGODB_DB_NAME][TEST_WRITES_COLLECTION].drop()


async def _thread_isolation(client, saver) -> None:
    thread_a = f"char-iso-a-{uuid.uuid4()}"
    thread_b = f"char-iso-b-{uuid.uuid4()}"
    await saver.aput(
        _thread_config(thread_a),
        _make_checkpoint(channel_values={"messages": ["A"]}),
        _make_metadata(),
        new_versions={},
    )
    await saver.aput(
        _thread_config(thread_b),
        _make_checkpoint(channel_values={"messages": ["B"]}),
        _make_metadata(),
        new_versions={},
    )

    loaded_a = await saver.aget_tuple(_thread_config(thread_a))
    loaded_b = await saver.aget_tuple(_thread_config(thread_b))
    assert loaded_a.checkpoint["channel_values"]["messages"] == ["A"]
    assert loaded_b.checkpoint["channel_values"]["messages"] == ["B"]


async def _hitl_pending_sends_persisted(client, saver) -> None:
    thread_id = f"char-hitl-{uuid.uuid4()}"
    pending = [
        ("call-1", {"type": "order_info_request", "missing_fields": ["partId"]}),
        ("call-2", {"type": "approval", "action_requests": []}),
    ]
    checkpoint = _make_checkpoint(
        channel_values={"messages": ["need input"]}, pending_sends=pending
    )
    try:
        await saver.aput(_thread_config(thread_id), checkpoint, _make_metadata(), new_versions={})
        client2 = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        saver2 = MongoDBSaver(
            client=client2,
            db_name=MONGODB_DB_NAME,
            checkpoint_collection_name=TEST_COLLECTION,
            writes_collection_name=TEST_WRITES_COLLECTION,
        )
        loaded = await saver2.aget_tuple(_thread_config(thread_id))
        assert loaded is not None
        # MongoDB 序列化后 tuple 变 list，值与结构等价
        assert [tuple(item) for item in loaded.checkpoint["pending_sends"]] == pending
        client2.close()
    finally:
        client[MONGODB_DB_NAME][TEST_COLLECTION].drop()
        client[MONGODB_DB_NAME][TEST_WRITES_COLLECTION].drop()


def test_roundtrip_checkpoint_content(client):
    """checkpoint 写入后可读回，内容一致（channel_values 与 pending_sends）。"""
    saver = MongoDBSaver(
        client=client,
        db_name=MONGODB_DB_NAME,
        checkpoint_collection_name=TEST_COLLECTION,
        writes_collection_name=TEST_WRITES_COLLECTION,
    )
    try:
        asyncio.run(_roundtrip(client, saver))
    finally:
        client[MONGODB_DB_NAME][TEST_COLLECTION].drop()
        client[MONGODB_DB_NAME][TEST_WRITES_COLLECTION].drop()


def test_restart_recovery_with_fresh_saver(client):
    """新 Saver 实例（模拟进程重启）能恢复同一 thread 的 checkpoint。"""
    asyncio.run(_restart_recovery(client))


def test_thread_isolation(client):
    """不同 thread 的 checkpoint 互不可见。"""
    saver = MongoDBSaver(
        client=client,
        db_name=MONGODB_DB_NAME,
        checkpoint_collection_name=TEST_COLLECTION,
        writes_collection_name=TEST_WRITES_COLLECTION,
    )
    try:
        asyncio.run(_thread_isolation(client, saver))
    finally:
        client[MONGODB_DB_NAME][TEST_COLLECTION].drop()
        client[MONGODB_DB_NAME][TEST_WRITES_COLLECTION].drop()


def test_hitl_pending_sends_persisted_across_restart(client):
    """HITL 中断状态（pending_sends）随 checkpoint 持久化，重启后可恢复。"""
    saver = MongoDBSaver(
        client=client,
        db_name=MONGODB_DB_NAME,
        checkpoint_collection_name=TEST_COLLECTION,
        writes_collection_name=TEST_WRITES_COLLECTION,
    )
    asyncio.run(_hitl_pending_sends_persisted(client, saver))
