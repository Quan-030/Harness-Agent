# src/test/test_char_display_messages.py
"""展示消息（session_display_messages）链路 characterization tests。

Memory v2 升级期间必须保留的行为（方案 21.3 第 1 步）：
- save_display_messages / get_display_messages round-trip 与顺序
- 整批替换语义（第二次 save 覆盖第一次）
- 超长字段截断（500KB）
- 不存在 thread 返回 None
- thread 隔离

注意：展示消息是面向会话历史展示的原始消息副本，不是长期记忆事实源
（方案 19.4 节），此行为在 Memory v2 中保持不变。

测试只写测试专属 thread_id 的数据，teardown 时清理，不触碰生产会话。
与仓库现有测试一致：同步测试函数 + 内部 asyncio.run（无 pytest-asyncio）。
"""
import asyncio
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api_view.agent_loader import AgentLoader, agent_loader
from agent.config import MONGODB_DB_NAME, MONGODB_URI
from pymongo import MongoClient


@pytest.fixture(autouse=True)
def mongo_client():
    """给全局 AgentLoader 单例注入 MongoDB client。

    AgentLoader._mongodb_client 由 initialize() 懒初始化（含沙箱预热/MCP 预计算，
    不适合测试）。save/get_display_messages 只依赖该 client，直接注入即可。
    """
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    agent_loader._mongodb_client = client
    yield client
    agent_loader._mongodb_client = None
    client.close()


@pytest.fixture()
def cleanup():
    thread_ids = []

    def _register(thread_id: str):
        thread_ids.append(thread_id)
        return thread_id

    yield _register
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    collection = client[MONGODB_DB_NAME]["session_display_messages"]
    collection.delete_many({"thread_id": {"$in": thread_ids}})
    client.close()


async def _roundtrip_preserves_order_and_content(cleanup) -> None:
    """save 后 get 返回同样的消息列表，顺序按写入顺序（index 升序）。"""
    thread_id = cleanup(f"char-dm-{uuid.uuid4()}")
    messages = [
        {"id": "user-1", "role": "user", "content": "分析博世报价"},
        {"id": "assistant-1", "role": "assistant", "content": "正在分析"},
        {"id": "tool-1", "role": "tool", "tool_name": "query_supplier", "tool_status": "done"},
    ]

    ok = await agent_loader.save_display_messages(thread_id, messages)
    assert ok is True

    loaded = await agent_loader.get_display_messages(thread_id)
    assert loaded == messages


async def _save_replaces_previous_batch(cleanup) -> None:
    """整批替换语义：第二次 save 完全覆盖第一次的内容。"""
    thread_id = cleanup(f"char-dm-{uuid.uuid4()}")
    first = [{"id": "user-1", "role": "user", "content": "旧问题"}]
    second = [
        {"id": "user-2", "role": "user", "content": "新问题"},
        {"id": "assistant-2", "role": "assistant", "content": "新回答"},
    ]

    await agent_loader.save_display_messages(thread_id, first)
    await agent_loader.save_display_messages(thread_id, second)

    loaded = await agent_loader.get_display_messages(thread_id)
    assert loaded == second


async def _get_unknown_thread_returns_none() -> None:
    """不存在的 thread 返回 None（不是空列表）。"""
    loaded = await agent_loader.get_display_messages(f"char-dm-none-{uuid.uuid4()}")
    assert loaded is None


async def _empty_save_clears_thread(cleanup) -> None:
    """保存空列表等价于清空该 thread 的展示消息。"""
    thread_id = cleanup(f"char-dm-{uuid.uuid4()}")
    await agent_loader.save_display_messages(
        thread_id, [{"id": "user-1", "role": "user", "content": "x"}]
    )
    await agent_loader.save_display_messages(thread_id, [])

    loaded = await agent_loader.get_display_messages(thread_id)
    assert loaded is None


async def _long_fields_are_truncated(cleanup) -> None:
    """超过 500KB 的 text/content/args 字段被截断并追加标记。"""
    thread_id = cleanup(f"char-dm-{uuid.uuid4()}")
    huge = "x" * (AgentLoader._MAX_FIELD_LENGTH + 1000)
    messages = [{"id": "tool-1", "role": "tool", "content": huge}]

    await agent_loader.save_display_messages(thread_id, messages)
    loaded = await agent_loader.get_display_messages(thread_id)
    assert loaded is not None
    marker = "\n\n...(内容过长已截断)"
    assert len(loaded[0]["content"]) == AgentLoader._MAX_FIELD_LENGTH + len(marker)
    assert loaded[0]["content"].endswith(marker)


async def _threads_are_isolated(cleanup) -> None:
    """不同 thread 的展示消息互不可见。"""
    thread_a = cleanup(f"char-dm-a-{uuid.uuid4()}")
    thread_b = cleanup(f"char-dm-b-{uuid.uuid4()}")
    await agent_loader.save_display_messages(
        thread_a, [{"id": "u1", "role": "user", "content": "A"}]
    )
    await agent_loader.save_display_messages(
        thread_b, [{"id": "u2", "role": "user", "content": "B"}]
    )

    loaded_a = await agent_loader.get_display_messages(thread_a)
    loaded_b = await agent_loader.get_display_messages(thread_b)
    assert loaded_a == [{"id": "u1", "role": "user", "content": "A"}]
    assert loaded_b == [{"id": "u2", "role": "user", "content": "B"}]


def test_roundtrip_preserves_order_and_content(cleanup):
    asyncio.run(_roundtrip_preserves_order_and_content(cleanup))


def test_save_replaces_previous_batch(cleanup):
    asyncio.run(_save_replaces_previous_batch(cleanup))


def test_get_unknown_thread_returns_none():
    asyncio.run(_get_unknown_thread_returns_none())


def test_empty_save_clears_thread(cleanup):
    asyncio.run(_empty_save_clears_thread(cleanup))


def test_long_fields_are_truncated(cleanup):
    asyncio.run(_long_fields_are_truncated(cleanup))


def test_threads_are_isolated(cleanup):
    asyncio.run(_threads_are_isolated(cleanup))
