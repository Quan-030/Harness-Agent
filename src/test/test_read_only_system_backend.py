# src/test/test_read_only_system_backend.py
"""ReadOnlyStoreBackend 测试：系统技能路由只读。

验证 /persisted-skills/system/ 背后的 backend 拒绝一切写入（含 async 路径），
但读取正常；同时 SkillsSyncMiddleware 走的 store.put 直写不受影响。

使用 langgraph InMemoryStore，不依赖 MongoDB。
"""
import asyncio
import sys
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.read_only_backend import ReadOnlyStoreBackend

NS = ("system_skills",)


class _FakeRuntime:
    """StoreBackend 只用到 runtime.store 和 runtime.state。"""

    def __init__(self, store):
        self.store = store
        self.state = None


@pytest.fixture()
def backend_and_store():
    store = InMemoryStore()
    backend = ReadOnlyStoreBackend(_FakeRuntime(store), namespace=lambda ctx: NS)
    return backend, store


def _seed(store, key="/main/demo/SKILL.md", content="hello"):
    store.put(NS, key, {"content": [content], "created_at": "t0", "modified_at": "t0"})


def test_sync_write_rejected_and_not_persisted(backend_and_store):
    backend, store = backend_and_store
    res = backend.write("/main/evil/SKILL.md", "pwned")
    assert res.error  # 返回错误结果
    assert res.path is None
    assert store.get(NS, "/main/evil/SKILL.md") is None  # 没写进去


def test_async_write_rejected_and_not_persisted(backend_and_store):
    backend, store = backend_and_store
    res = asyncio.run(backend.awrite("/main/evil/SKILL.md", "pwned"))
    assert res.error
    assert store.get(NS, "/main/evil/SKILL.md") is None


def test_sync_edit_rejected_keeps_original(backend_and_store):
    backend, store = backend_and_store
    _seed(store)
    res = backend.edit("/main/demo/SKILL.md", "hello", "tampered")
    assert res.error
    assert store.get(NS, "/main/demo/SKILL.md").value["content"] == ["hello"]


def test_async_edit_rejected_keeps_original(backend_and_store):
    backend, store = backend_and_store
    _seed(store)
    res = asyncio.run(backend.aedit("/main/demo/SKILL.md", "hello", "tampered"))
    assert res.error
    assert store.get(NS, "/main/demo/SKILL.md").value["content"] == ["hello"]


def test_upload_files_rejected(backend_and_store):
    backend, store = backend_and_store
    responses = backend.upload_files([("/main/evil/SKILL.md", b"pwned")])
    assert all(r.error == "permission_denied" for r in responses)
    assert store.get(NS, "/main/evil/SKILL.md") is None


def test_read_still_works(backend_and_store):
    backend, store = backend_and_store
    _seed(store, content="readable")
    out = backend.read("/main/demo/SKILL.md")
    assert "readable" in out  # 读取正常
    assert "只读" not in out  # 不是只读错误


def test_direct_store_put_still_works(backend_and_store):
    """代表 SkillsSyncMiddleware：直写 store 不经过 backend，应正常。"""
    _, store = backend_and_store
    store.put(NS, "/main/sys/SKILL.md", {"content": ["sys"], "created_at": "t0", "modified_at": "t0"})
    assert store.get(NS, "/main/sys/SKILL.md").value["content"] == ["sys"]
