# src/test/memory/test_api.py
"""Memory v2 最小 API 集成测试（方案 20.2）。

使用独立测试库 memory_v2_api_test。端点函数直接异步调用
（不使用 TestClient：其内部 event loop 与 asyncmy 连接池不兼容）。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.database import memory_database  # noqa: E402
from agent.memory.models import MemoryKind  # noqa: E402
from agent.memory.policies import MemoryPolicy  # noqa: E402
from agent.memory.repository import (  # noqa: E402
    ActorContext,
    MySQLMemoryRepository,
)
from agent.memory.service import MemoryService  # noqa: E402
from api_view.api import memory as memory_api  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_api_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

# async engine 连接池绑定创建它的 event loop：全模块共享一个 loop，
# 避免 fixture 与测试各自 asyncio.run 导致跨 loop 使用连接
LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)


def _loop_run(coro):
    return LOOP.run_until_complete(coro)


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    """session 级：建库 + upgrade；teardown 时删除并确保全局单例已 dispose。"""
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

    _loop_run(_create())
    cfg = Config(ALEMBIC_INI)
    os.environ["MEMORY_MYSQL_DSN"] = DSN
    command.upgrade(cfg, "head")
    yield
    _loop_run(memory_database.dispose())
    _loop_run(_drop())


@pytest.fixture()
def db():
    """初始化全局 memory_database（指向测试库），teardown 时 dispose。"""
    async def _init():
        await memory_database.initialize(
            dsn=DSN,
            pool_size=5,
            pool_max_overflow=5,
            connect_timeout=5,
            expected_revision="0002",
            write_enabled=True,
            read_enabled=True,
            jobs_enabled=False,
            semantic_enabled=False,
        )

    _loop_run(_init())
    yield
    _loop_run(memory_database.dispose())


def _make_request(headers: dict | None = None) -> Request:
    h = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request(
        scope={
            "type": "http",
            "method": "PATCH",
            "path": "/api/memory/profile",
            "query_string": b"",
            "headers": h,
        }
    )


def _service() -> MemoryService:
    return MemoryService(
        MySQLMemoryRepository(memory_database.session_factory), MemoryPolicy()
    )


async def _seed(user: str, content: str = "博世询价成功") -> str:
    result = await _service().remember(
        user,
        ActorContext(actor_type="user", source_thread_id="test-thread"),
        kind=MemoryKind.TASK_OUTCOME,
        content=content,
        data={"task_type": "询价", "result_status": "succeeded"},
    )
    return result.memory_id


# ============================================================
# Profile
# ============================================================

def test_get_profile_auto_creates(db):
    async def _run():
        return await memory_api.get_profile(user_id="api-user-1")

    body = _loop_run(_run())
    assert body["user_id"] == "api-user-1"
    assert body["version"] == 1


def test_patch_profile_missing_version_428(db):
    patch = {"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}]}

    async def _run():
        try:
            await memory_api.patch_profile(
                _make_request(), memory_api.ProfilePatchRequest(user_id="api-user-1", patch=patch)
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    assert _loop_run(_run()) == 428


def test_patch_profile_with_version(db):
    patch = {"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}]}

    async def _run():
        await memory_api.get_profile(user_id="api-user-2")
        return await memory_api.patch_profile(
            _make_request(),
            memory_api.ProfilePatchRequest(
                user_id="api-user-2", patch=patch, expected_version=1
            ),
        )

    body = _loop_run(_run())
    assert body["currency"] == "CNY"
    assert body["version"] == 2


def test_patch_profile_if_match_header(db):
    patch = {"scalar_ops": [{"field": "language", "op": "set", "value": "zh"}]}

    async def _run():
        await memory_api.get_profile(user_id="api-user-3")
        return await memory_api.patch_profile(
            _make_request({"If-Match": "1"}),
            memory_api.ProfilePatchRequest(user_id="api-user-3", patch=patch),
        )

    body = _loop_run(_run())
    assert body["language"] == "zh"


def test_patch_profile_version_conflict_409(db):
    patch = {"scalar_ops": [{"field": "currency", "op": "set", "value": "USD"}]}

    async def _run():
        await memory_api.get_profile(user_id="api-user-4")
        await memory_api.patch_profile(
            _make_request(),
            memory_api.ProfilePatchRequest(
                user_id="api-user-4", patch=patch, expected_version=1
            ),
        )
        try:
            await memory_api.patch_profile(
                _make_request(),
                memory_api.ProfilePatchRequest(
                    user_id="api-user-4", patch=patch, expected_version=1
                ),
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    assert _loop_run(_run()) == 409


# ============================================================
# Memories
# ============================================================

def test_list_memories_empty(db):
    async def _run():
        return await memory_api.list_memories(
            user_id="api-list-1", kind=None, status=None, cursor=None, limit=50
        )

    body = _loop_run(_run())
    assert body["items"] == []


def test_remember_then_list_correct_and_forget(db):
    user = "api-crud-1"

    async def _run():
        memory_id = await _seed(user)
        # 列表
        page = await memory_api.list_memories(
            user_id=user, kind=None, status=None, cursor=None, limit=50
        )
        assert len(page["items"]) == 1
        assert page["items"][0]["content"] == "博世询价成功"
        # 纠正：旧条目 superseded，新条目 active
        corrected = await memory_api.correct_memory(
            memory_id,
            memory_api.CorrectMemoryRequest(user_id=user, content="博世询价结果已修正"),
        )
        assert corrected["outcome"] == "superseded"
        # 删除新条目（旧条目已 superseded 不在默认列表）
        await memory_api.forget_memory(
            corrected["memory_id"], memory_api.ForgetRequest(user_id=user)
        )
        page2 = await memory_api.list_memories(
            user_id=user, kind=None, status=None, cursor=None, limit=50
        )
        return len(page2["items"])

    assert _loop_run(_run()) == 0


def test_delete_unknown_memory_404(db):
    async def _run():
        try:
            await memory_api.forget_memory(
                "00000000-0000-7000-8000-000000000000",
                memory_api.ForgetRequest(user_id="api-none-1"),
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    assert _loop_run(_run()) == 404


# ============================================================
# delete-all（二次确认）
# ============================================================

def test_delete_all_prepare_and_confirm(db):
    user = "api-delete-all-1"

    async def _run():
        await _seed(user)
        prepare = await memory_api.delete_all_prepare(
            memory_api.UserScope(user_id=user)
        )
        token = prepare["token"]
        confirm = await memory_api.delete_all_confirm(
            memory_api.DeleteAllConfirmRequest(token=token)
        )
        assert confirm["deleted"] is True
        # token 一次性：复用 → 400
        try:
            await memory_api.delete_all_confirm(
                memory_api.DeleteAllConfirmRequest(token=token)
            )
        except HTTPException as exc:
            assert exc.status_code == 400
        # 删除后列表为空
        page = await memory_api.list_memories(
            user_id=user, kind=None, status=None, cursor=None, limit=50
        )
        return len(page["items"])

    assert _loop_run(_run()) == 0


def test_delete_all_invalid_token_400(db):
    async def _run():
        try:
            await memory_api.delete_all_confirm(
                memory_api.DeleteAllConfirmRequest(token="invalid-token-xyz")
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    assert _loop_run(_run()) == 400


# ============================================================
# capability 门控（review #8 / I）
# ============================================================

def test_all_disabled_endpoints_503():
    """READ=0 WRITE=0：读/写端点均 503（完全关闭）。"""
    async def _init():
        await memory_database.initialize(
            dsn=DSN,
            pool_size=5,
            pool_max_overflow=5,
            connect_timeout=5,
            expected_revision="0002",
            write_enabled=False,
            read_enabled=False,
            jobs_enabled=False,
            semantic_enabled=False,
        )

    async def _run_read():
        try:
            await memory_api.get_profile(user_id="x")
        except HTTPException as exc:
            return exc.status_code
        return None

    async def _run_write():
        try:
            await memory_api.patch_profile(
                _make_request(),
                memory_api.ProfilePatchRequest(
                    user_id="x",
                    patch={"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}]},
                ),
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    _loop_run(_init())
    try:
        assert _loop_run(_run_read()) == 503
        assert _loop_run(_run_write()) == 503
    finally:
        _loop_run(memory_database.dispose())


def test_prewarm_state_read_disabled_write_enabled():
    """WRITE=1 READ=0（预热状态）：读端点 503，写端点可用（review #8）。"""
    async def _init():
        await memory_database.initialize(
            dsn=DSN,
            pool_size=5,
            pool_max_overflow=5,
            connect_timeout=5,
            expected_revision="0002",
            write_enabled=True,
            read_enabled=False,
            jobs_enabled=False,
            semantic_enabled=False,
        )

    async def _run_read():
        try:
            await memory_api.get_profile(user_id="x")
        except HTTPException as exc:
            return exc.status_code
        return None

    async def _run_write():
        try:
            # 预热状态允许显式写入：先经 service 建 profile（内部路径），
            # 再走 API PATCH（带版本）验证写门控放行
            await memory_api._service().get_profile("x")
            await memory_api.patch_profile(
                _make_request(),
                memory_api.ProfilePatchRequest(
                    user_id="x",
                    expected_version=1,
                    patch={"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}]},
                ),
            )
        except HTTPException as exc:
            return exc.status_code
        return None

    _loop_run(_init())
    try:
        assert _loop_run(_run_read()) == 503
        assert _loop_run(_run_write()) is None  # 写路径不抛（预热状态允许显式写入）
    finally:
        _loop_run(memory_database.dispose())


# ============================================================
# 503：Memory v2 未初始化
# ============================================================

def test_api_503_when_memory_not_initialized():
    """memory_database 未初始化（v2 关闭）时 API 返回 503。"""
    async def _run():
        try:
            await memory_api.get_profile(user_id="x")
        except HTTPException as exc:
            return exc.status_code
        return None

    _loop_run(memory_database.dispose())
    assert _loop_run(_run()) == 503
