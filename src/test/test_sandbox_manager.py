"""
sandbox_manager 单元测试 — 预热沙箱认领逻辑。

不依赖真实 MongoDB / OpenSandbox：
- agent.config 与 agent.backends.sandbox_setup 在导入前用存根替换
- MongoDB 集合用 FakeCollection 模拟
- 沙箱 backend 用 FakeBackend 模拟
"""
import asyncio
import sys
import types

import pytest

# ---- 在导入 sandbox_manager 前安装存根，切断 agent.config 的急切外连 ----

_config_stub = types.ModuleType("agent.config")
_config_stub.SANDBOX_CONFIG = object()
_config_stub.MONGODB_DB_NAME = "testdb"
sys.modules.setdefault("agent.config", _config_stub)

_setup_stub = types.ModuleType("agent.backends.sandbox_setup")
_setup_stub.setup_sandbox = None  # 每个测试用 monkeypatch 注入
sys.modules.setdefault("agent.backends.sandbox_setup", _setup_stub)

from agent.backends import sandbox_manager as sm  # noqa: E402
from agent.backends.sandbox_proxy import SandboxBackendProxy  # noqa: E402


# ---- 测试替身 ----


class FakeBackend:
    """模拟 OpenSandboxBackend：只需 id 和 execute。"""

    def __init__(self, backend_id: str, alive: bool = True):
        self._id = backend_id
        self.alive = alive

    @property
    def id(self) -> str:
        return self._id

    def execute(self, command, *args, **kwargs):
        if not self.alive:
            raise RuntimeError(f"sandbox {self._id} dead")
        return "ok"


class FakeCollection:
    """模拟 sandbox_registry 集合：按 user_id 存单文档。"""

    def __init__(self):
        self.docs: dict[str, dict] = {}

    def find_one(self, flt):
        return self.docs.get(flt["user_id"])

    def update_one(self, flt, update, upsert=False):
        user_id = flt["user_id"]
        if user_id not in self.docs:
            if not upsert:
                return
            self.docs[user_id] = {"user_id": user_id}
        self.docs[user_id].update(update["$set"])

    def delete_one(self, flt):
        self.docs.pop(flt["user_id"], None)

    def create_index(self, *args, **kwargs):
        pass


@pytest.fixture()
def env(monkeypatch):
    """隔离 sandbox_manager 全局状态：假集合、禁用后台补充、拦截沙箱删除。"""
    coll = FakeCollection()
    monkeypatch.setattr(sm, "_sandbox_collection", lambda: coll)
    monkeypatch.setattr(sm, "SANDBOX_BACKENDS", {})

    async def _noop_replenish():
        pass

    monkeypatch.setattr(sm, "_replenish_warm", _noop_replenish)

    deleted: list[str] = []
    if hasattr(sm, "_try_delete_sandbox"):
        async def _fake_delete(sandbox_id):
            deleted.append(sandbox_id)

        monkeypatch.setattr(sm, "_try_delete_sandbox", _fake_delete)
    coll.deleted = deleted

    sm._warm_reserve = None
    yield coll
    sm._warm_reserve = None


def _patch_setup(monkeypatch, fn):
    monkeypatch.setattr(_setup_stub, "setup_sandbox", fn)


# ---- 缺陷 2：老用户应优先重连自己的沙箱，而不是认领预热沙箱 ----


def test_returning_user_reconnects_own_sandbox_instead_of_claiming_warm(
    env, monkeypatch,
):
    coll = env
    coll.docs["u1"] = {"user_id": "u1", "sandbox_id": "old-sb"}
    sm._warm_reserve = SandboxBackendProxy(FakeBackend("warm-sb"))

    def fake_setup(config, sandbox_id=None, image=None):
        assert sandbox_id == "old-sb", "应尝试重连老沙箱而不是新建"
        return FakeBackend(sandbox_id)

    _patch_setup(monkeypatch, fake_setup)

    proxy = asyncio.run(sm.ensure_sandbox_for_user("u1"))

    assert proxy.id == "old-sb"
    assert sm._warm_reserve is not None, "预热沙箱不应被老用户消耗"
    assert sm._warm_reserve.id == "warm-sb"
    assert coll.docs["u1"]["sandbox_id"] == "old-sb"


# ---- 缺陷 1：认领预热沙箱前必须健康检查，死沙箱应被丢弃 ----


def test_dead_warm_reserve_is_discarded_and_new_sandbox_created(
    env, monkeypatch,
):
    coll = env
    sm._warm_reserve = SandboxBackendProxy(FakeBackend("warm-dead", alive=False))

    def fake_setup(config, sandbox_id=None, image=None):
        assert sandbox_id is None
        return FakeBackend("fresh-sb")

    _patch_setup(monkeypatch, fake_setup)

    proxy = asyncio.run(sm.ensure_sandbox_for_user("u2"))

    assert proxy.id == "fresh-sb", "不应把死掉的预热沙箱分配给用户"
    assert sm._warm_reserve is None, "死掉的预热沙箱应被丢弃"
    assert coll.docs["u2"]["sandbox_id"] == "fresh-sb"
    assert "warm-dead" in coll.deleted, "死掉的预热沙箱应尝试删除"


# ---- 回归保护：新用户认领健康预热沙箱的原有行为不变 ----


def test_new_user_claims_healthy_warm_reserve(env, monkeypatch):
    coll = env
    sm._warm_reserve = SandboxBackendProxy(FakeBackend("warm-sb"))

    def fake_setup(config, sandbox_id=None, image=None):
        raise AssertionError("有健康预热沙箱时不应新建")

    _patch_setup(monkeypatch, fake_setup)

    proxy = asyncio.run(sm.ensure_sandbox_for_user("u3"))

    assert proxy.id == "warm-sb"
    assert sm._warm_reserve is None
    assert coll.docs["u3"]["sandbox_id"] == "warm-sb"


# ---- 行为规格：老用户重连失败时，可认领健康预热沙箱作为替代 ----


def test_returning_user_with_dead_sandbox_falls_back_to_warm_reserve(
    env, monkeypatch,
):
    coll = env
    coll.docs["u4"] = {"user_id": "u4", "sandbox_id": "gone-sb"}
    sm._warm_reserve = SandboxBackendProxy(FakeBackend("warm-sb"))

    def fake_setup(config, sandbox_id=None, image=None):
        if sandbox_id == "gone-sb":
            raise RuntimeError("sandbox destroyed")
        raise AssertionError("有健康预热沙箱时不应新建")

    _patch_setup(monkeypatch, fake_setup)

    proxy = asyncio.run(sm.ensure_sandbox_for_user("u4"))

    assert proxy.id == "warm-sb"
    assert sm._warm_reserve is None
    assert coll.docs["u4"]["sandbox_id"] == "warm-sb"
