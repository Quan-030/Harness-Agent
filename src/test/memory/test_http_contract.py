# src/test/memory/test_http_contract.py
"""Memory v2 HTTP API contract 集成测试（review K/L）。

使用 httpx.ASGITransport 在同一 event loop 内跑真实 ASGI 链路
（routing / body/query validation / exception handler / HTTP status /
header / serialization），并验证 API contract：
- 请求校验失败 → 400（非 422）
- 非法 cursor → 400（非 500）
- Memory v2 未启用 → 503
"""
import asyncio
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402

from agent.memory.database import memory_database  # noqa: E402
from agent.memory.models import MemoryKind  # noqa: E402
from api_view.api import memory as memory_api  # noqa: E402

BASE_URL = os.getenv(
    "MEMORY_MYSQL_BASE_URL",
    "mysql+asyncmy://root:123456@127.0.0.1:3306",
)
TEST_DB = "memory_v2_http_test"
DSN = f"{BASE_URL}/{TEST_DB}"
ALEMBIC_INI = str(Path(__file__).parent.parent.parent.parent / "alembic.ini")

# review 3.5：测试与生产共用同一 handler 注册逻辑（从 web_main 导入）
from api_view.web_main import register_memory_exception_handlers  # noqa: E402


def _build_app() -> FastAPI:
    app = FastAPI()
    register_memory_exception_handlers(app)
    app.include_router(memory_api.router, prefix="/api")

    # dummy 非 memory 路由（review 3.4：验证其他路由保持 FastAPI 默认 422）
    from pydantic import BaseModel

    class _ChatBody(BaseModel):
        message: str

    @app.post("/api/chat/test")
    async def _dummy_chat(body: _ChatBody):
        return {"ok": True}

    return app


@pytest.fixture(scope="session", autouse=True)
def migrated_db():
    """session 级：建库 + upgrade；teardown 时删除并 dispose 全局单例。"""
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
    asyncio.run(memory_database.dispose())
    asyncio.run(_drop())


@pytest.fixture()
def http_client():
    """初始化全局 memory_database 后返回 ASGITransport AsyncClient（同一 loop）。"""
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_init())
    app = _build_app()

    async def _make_client():
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        )

    client = loop.run_until_complete(_make_client())
    yield loop, client
    loop.run_until_complete(client.aclose())
    loop.run_until_complete(memory_database.dispose())
    loop.close()


# ============================================================
# 真实 ASGI 链路
# ============================================================

def test_get_profile_via_asgi(http_client):
    loop, client = http_client
    resp = loop.run_until_complete(
        client.get("/api/memory/profile", params={"user_id": "http-user-1"})
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "http-user-1"
    assert body["version"] == 1


def test_validation_error_maps_to_400(http_client):
    """user_id 为空 → RequestValidationError → 400（review K，非 422）。"""
    loop, client = http_client
    resp = loop.run_until_complete(
        client.get("/api/memory/profile", params={"user_id": ""})
    )
    assert resp.status_code == 400


def test_invalid_cursor_maps_to_400(http_client):
    """非法 cursor（非 base64）→ 400（review L，非 500）。"""
    loop, client = http_client
    resp = loop.run_until_complete(
        client.get("/api/memories", params={"user_id": "http-u2", "cursor": "!!!not-base64!!!"})
    )
    assert resp.status_code == 400


def test_invalid_cursor_shape_maps_to_400(http_client):
    """合法 base64 但形状非法 → 400（review L）。"""
    loop, client = http_client
    import base64 as b64
    bad = b64.urlsafe_b64encode(b"no-pipe-separator").decode("ascii")
    resp = loop.run_until_complete(
        client.get("/api/memories", params={"user_id": "http-u3", "cursor": bad})
    )
    assert resp.status_code == 400


def test_patch_profile_flow_via_asgi(http_client):
    """PATCH profile 全链路：版本缺失 428 → 带版本 200 → 冲突 409。"""
    loop, client = http_client
    patch = {
        "user_id": "http-u4",
        "patch": {"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}]},
    }
    # 无版本 → 428
    r1 = loop.run_until_complete(client.patch("/api/memory/profile", json=patch))
    assert r1.status_code == 428
    # 先 GET 创建 profile，再带版本 PATCH → 200
    loop.run_until_complete(
        client.get("/api/memory/profile", params={"user_id": "http-u4"})
    )
    patch["expected_version"] = 1
    r2 = loop.run_until_complete(client.patch("/api/memory/profile", json=patch))
    assert r2.status_code == 200
    assert r2.json()["currency"] == "CNY"
    # 旧版本再改 → 409
    r3 = loop.run_until_complete(client.patch("/api/memory/profile", json=patch))
    assert r3.status_code == 409


def test_delete_all_flow_via_asgi(http_client):
    """delete-all prepare/confirm 全链路（token 校验/范围）。"""
    loop, client = http_client
    r1 = loop.run_until_complete(
        client.post("/api/memories/delete-all/prepare", json={"user_id": "http-u5"})
    )
    assert r1.status_code == 200
    token = r1.json()["token"]
    assert "scope" in r1.json()

    r2 = loop.run_until_complete(
        client.post("/api/memories/delete-all/confirm", json={"token": token})
    )
    assert r2.status_code == 200
    assert r2.json()["deleted"] is True

    r3 = loop.run_until_complete(
        client.post("/api/memories/delete-all/confirm", json={"token": token})
    )
    assert r3.status_code == 400  # token 一次性


# ============================================================
# review 3.2：领域错误映射（400/409，非 500）
# ============================================================

def test_correct_memory_invalid_data_400(http_client):
    """PATCH 纠正时 data 不符合 kind 契约 → 400（review 3.2）。"""
    loop, client = http_client
    user = "http-dom-1"

    async def _seed() -> str:
        from agent.memory.policies import MemoryPolicy
        from agent.memory.repository import ActorContext, MySQLMemoryRepository
        from agent.memory.service import MemoryService

        service = MemoryService(
            MySQLMemoryRepository(memory_database.session_factory), MemoryPolicy()
        )
        result = await service.remember(
            user,
            ActorContext(actor_type="user", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="原始任务结果",
            data={"task_type": "询价", "result_status": "succeeded"},
        )
        return result.memory_id

    memory_id = loop.run_until_complete(_seed())
    # data={} 对 task_outcome 缺必填字段 → 400
    resp = loop.run_until_complete(
        client.patch(
            f"/api/memories/{memory_id}",
            json={"user_id": user, "content": "修正内容", "data": {}},
        )
    )
    assert resp.status_code == 400


def test_correct_superseded_memory_409(http_client):
    """纠正已 superseded 的记忆 → 409（review 3.2）。"""
    loop, client = http_client
    user = "http-dom-2"

    async def _seed() -> str:
        from agent.memory.policies import MemoryPolicy
        from agent.memory.repository import ActorContext, MySQLMemoryRepository
        from agent.memory.service import MemoryService

        service = MemoryService(
            MySQLMemoryRepository(memory_database.session_factory), MemoryPolicy()
        )
        result = await service.remember(
            user,
            ActorContext(actor_type="user", source_thread_id="t1"),
            kind=MemoryKind.TASK_OUTCOME,
            content="原始任务结果",
            data={"task_type": "询价", "result_status": "succeeded"},
        )
        return result.memory_id

    memory_id = loop.run_until_complete(_seed())
    # 第一次纠正成功（旧条目 superseded）
    r1 = loop.run_until_complete(
        client.patch(
            f"/api/memories/{memory_id}",
            json={"user_id": user, "content": "第一次修正",
                  "data": {"task_type": "询价", "result_status": "failed"}},
        )
    )
    assert r1.status_code == 200
    # 再次纠正同一旧条目 → 409
    r2 = loop.run_until_complete(
        client.patch(
            f"/api/memories/{memory_id}",
            json={"user_id": user, "content": "第二次修正",
                  "data": {"task_type": "询价", "result_status": "failed"}},
        )
    )
    assert r2.status_code == 409


def test_patch_profile_final_state_invalid_400(http_client):
    """Profile 最终状态越界（49 + add 2 = 51）→ 400（review 3.2/N）。"""
    loop, client = http_client
    user = "http-dom-3"
    loop.run_until_complete(
        client.get("/api/memory/profile", params={"user_id": user})
    )
    # 直接塞 49 条 quality_standards（JSON 字符串参数，asyncmy 不支持 list 参数）
    import json
    from sqlalchemy import text as sql_text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _seed_49():
        engine = create_async_engine(DSN)
        async with engine.begin() as conn:
            items = [f"QS{i:02d}" for i in range(49)]
            await conn.execute(
                sql_text(
                    "UPDATE memory_profiles SET procurement_defaults = JSON_OBJECT("
                    "'quality_standards', CAST(:items AS JSON)) WHERE user_id = :u"
                ),
                {"u": user, "items": json.dumps(items)},
            )
        await engine.dispose()

    loop.run_until_complete(_seed_49())
    # add 2 条 → 51 条越界 → 400
    resp = loop.run_until_complete(
        client.patch(
            "/api/memory/profile",
            json={
                "user_id": user,
                "expected_version": 1,
                "patch": {"list_ops": [{"field": "quality_standards", "op": "add",
                                        "values": ["QS99", "QS100"]}]},
            },
        )
    )
    assert resp.status_code == 400


# ============================================================
# review 3.4：非 memory 路由保持 422
# ============================================================

def test_non_memory_route_keeps_422(http_client):
    """chat 等非 Memory 路由的校验错误保持 FastAPI 默认 422（review 3.4）。"""
    loop, client = http_client
    resp = loop.run_until_complete(
        client.post("/api/chat/test", json={"wrong_field": "x"})
    )
    assert resp.status_code == 422


# ============================================================
# review 3.5：真实 OperationalError → 503
# ============================================================

def test_runtime_db_unavailable_503(http_client, monkeypatch):
    """MySQL 运行中断开（OperationalError）→ 503，响应不含 SQL/DSN（review 3.5/K）。"""
    loop, client = http_client
    from sqlalchemy.exc import OperationalError as SAOperationalError

    def _boom(*args, **kwargs):
        # 同步函数直接抛（端点内 _require_read_service() 是同步调用）
        raise SAOperationalError(
            "stmt", {}, Exception("(2003, Can't connect to MySQL server on '127.0.0.1' (111))")
        )

    # 让读端点内部抛 OperationalError
    monkeypatch.setattr(memory_api, "_require_read_service", _boom)
    resp = loop.run_until_complete(
        client.get("/api/memories", params={"user_id": "http-503"})
    )
    assert resp.status_code == 503
    body = resp.text
    # 响应不含底层异常细节（IP/连接错误/参数），只含业务文案
    assert "127.0.0.1" not in body
    assert "Can't connect" not in body
    assert "(2003" not in body
