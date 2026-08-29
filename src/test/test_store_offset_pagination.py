# src/test/test_store_offset_pagination.py
"""OffsetCompatMongoDBStore 翻页回归测试。

deepagents 的 StoreBackend 通过 offset 翻页遍历 /memories/ 与 /persisted-skills/，
而原生 MongoDBStore 对 offset>0 抛 NotImplementedError。本测试复现 StoreBackend 的
分页循环（offset += 100），验证补丁后能完整、无重复地取回 >100 条数据且不崩溃。

使用真实 MongoDB 上的一次性 collection（store_offset_test_tmp），teardown drop。
"""
import os
import sys
from pathlib import Path

import pytest
from pymongo import MongoClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.store_compat import OffsetCompatMongoDBStore

MONGODB_URI = os.getenv(
    "MONGODB_URI",
    "mongodb://root:123456@127.0.0.1:27017/?authSource=admin",
)
TEST_DB = "langchain_db"
TEST_COLLECTION = "store_offset_test_tmp"


@pytest.fixture()
def store():
    client = MongoClient(MONGODB_URI)
    collection = client[TEST_DB][TEST_COLLECTION]
    yield OffsetCompatMongoDBStore(collection=collection)
    client[TEST_DB].drop_collection(TEST_COLLECTION)
    client.close()


def test_nonzero_offset_does_not_raise(store):
    """原生 MongoDBStore 在此处会抛 NotImplementedError。"""
    ns = ("system_skills",)
    for i in range(150):
        store.put(ns, f"/main/skill-{i:03d}/SKILL.md", {"content": [f"s{i}"]})

    page = store.search(ns, limit=100, offset=100)
    assert len(page) == 50  # 150 总数，跳过前 100 → 剩 50


def test_storebackend_style_pagination_collects_all(store):
    """模拟 StoreBackend._search_store_paginated：offset += 100 直到取空。"""
    ns = ("system_skills",)
    expected_keys = set()
    for i in range(250):
        key = f"/main/skill-{i:03d}/SKILL.md"
        expected_keys.add(key)
        store.put(ns, key, {"content": [f"s{i}"]})

    all_items = []
    offset = 0
    page_size = 100
    while True:
        page = store.search(ns, limit=page_size, offset=offset)
        if not page:
            break
        all_items.extend(page)
        if len(page) < page_size:
            break
        offset += page_size

    collected_keys = [item.key for item in all_items]
    assert len(collected_keys) == 250          # 无重复多取
    assert set(collected_keys) == expected_keys  # 无遗漏、全覆盖
