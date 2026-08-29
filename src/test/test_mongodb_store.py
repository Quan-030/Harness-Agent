# src/test/test_mongodb_store.py
"""MongoDBStore 持久化集成测试。

使用真实 MongoDB 实例上的一次性测试 collection（store_test_tmp），
teardown 时整个 collection 被 drop，不触碰生产 collection。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.store.mongodb import MongoDBStore

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://root:123456@127.0.0.1:27017/?authSource=admin",
)
TEST_DB = "langchain_db"
TEST_COLLECTION = "store_test_tmp"


@pytest.fixture()
def store():
    client = MongoClient(MONGODB_URI)
    collection = client[TEST_DB][TEST_COLLECTION]
    yield MongoDBStore(collection=collection)
    client[TEST_DB].drop_collection(TEST_COLLECTION)
    client.close()


def test_sync_put_get_search(store):
    ns = ("system_skills",)
    value = {"content": ["hello"], "created_at": "t0", "modified_at": "t0"}
    store.put(ns, "/main/demo/SKILL.md", value)

    item = store.get(ns, "/main/demo/SKILL.md")
    assert item is not None
    assert item.value["content"] == ["hello"]

    results = store.search(ns)
    assert any(i.key == "/main/demo/SKILL.md" for i in results)


def test_async_put_get_search(store):
    async def _run():
        ns = ("memories-roundtrip",)
        await store.aput(ns, "preferences.md", {"content": ["likes tea"]})

        item = await store.aget(ns, "preferences.md")
        assert item is not None
        assert item.value["content"] == ["likes tea"]

        results = await store.asearch(ns)
        assert any(i.key == "preferences.md" for i in results)

    asyncio.run(_run())


def test_collect_skills_restores_all_files_as_bytes(store):
    """恢复路径 bug 回归测试：asearch 默认 limit=10 截断；list 形态 content 未转 bytes。"""
    from agent.middlewares.user_skills_restore import UserSkillsRestoreMiddleware

    ns = ("system_skills",)
    for i in range(120):
        # 与 assign_skill._persist_skill 一致的存储格式：content 是单元素 list
        store.put(ns, f"/main/skill-{i:03d}/SKILL.md", {"content": [f"skill {i}"]})

    # StoreBackend 路由（/persisted-skills/）按行拆分存储：list of lines，换行符已剥离
    store.put(ns, "/main/multi-line/SKILL.md", {"content": ["line1", "line2"]})

    mw = UserSkillsRestoreMiddleware(None, system_namespace=ns, user_id="test_user")
    files = asyncio.run(mw._collect_skills(store, mw.system_namespace))

    assert len(files) == 121
    assert all(isinstance(content, bytes) for _, content in files)
    by_path = dict(files)
    assert by_path["/skills/main/skill-007/SKILL.md"] == b"skill 7"
    assert by_path["/skills/main/multi-line/SKILL.md"] == b"line1\nline2"


def test_is_system_skill_detects_existing_skill(store):
    """_is_system_skill 对系统命名空间中存在的技能返回 True。"""
    from agent.tools.assign_skill import _is_system_skill

    ns = ("system_skills",)
    store.put(ns, "/main/skill-management/SKILL.md", {"content": ["hello"]})
    store.put(ns, "/main/skill-management/scripts/download.py", {"content": ["print(1)"]})
    # 不相关技能不应干扰检测
    store.put(ns, "/main/other-skill/SKILL.md", {"content": ["other"]})

    result = asyncio.run(_is_system_skill(store, "skill-management"))
    assert result is True


def test_is_system_skill_returns_false_for_unknown_skill(store):
    """不存在于系统命名空间的技能返回 False。"""
    from agent.tools.assign_skill import _is_system_skill

    ns = ("system_skills",)
    store.put(ns, "/main/skill-management/SKILL.md", {"content": ["hello"]})

    result = asyncio.run(_is_system_skill(store, "nonexistent-skill"))
    assert result is False


def test_is_system_skill_handles_empty_namespace(store):
    """系统命名空间为空时返回 False（通过本地文件系统回退或返回 False）。"""
    from agent.tools.assign_skill import _is_system_skill

    # store fixture 是空的，直接测试
    result = asyncio.run(_is_system_skill(store, "any-skill"))
    assert result is False


def test_is_system_skill_matches_any_scope(store):
    """_is_system_skill 匹配任意 scope 下的技能（不限于 main）。"""
    from agent.tools.assign_skill import _is_system_skill

    ns = ("system_skills",)
    # system skills 可能存在于 procurement scope
    store.put(ns, "/procurement/web-scraper/SKILL.md", {"content": ["scraper"]})

    result = asyncio.run(_is_system_skill(store, "web-scraper"))
    assert result is True
