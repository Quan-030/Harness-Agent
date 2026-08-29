# src/api_view/api/memory.py
"""Memory v2 最小 API（方案 20.2）。

- GET    /api/memory/profile                 读取当前用户 Profile
- PATCH  /api/memory/profile                 If-Match/version 乐观锁更新
- GET    /api/memories                       cursor 分页、kind/status 过滤
- PATCH  /api/memories/{memory_id}           纠正内容（内部新 Item + supersede 旧 Item）
- DELETE /api/memories/{memory_id}           标记 forgotten
- POST   /api/memories/delete-all/prepare    返回短期确认 token 和删除范围
- POST   /api/memories/delete-all/confirm    执行确认后的删除

状态码：400 请求/字段校验失败；404 scope 内不存在；409 版本/并发冲突；
428 缺少 Profile 版本；503 MySQL 暂不可用（Memory v2 未初始化）。
user_id 是功能性 scope，不是认证身份（方案 8.1）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from agent.memory.database import memory_database
from agent.memory.models import MemoryKind, MemoryStatus, ProfilePatch
from agent.memory.policies import MemoryPolicy, SensitiveContentError
from agent.memory.repository import (
    ActorContext,
    InvalidMemoryCursor,
    InvalidMemoryData,
    InvalidMemoryState,
    InvalidProfilePatch,
    MemoryListFilter,
    MemoryNotFound,
    MySQLMemoryRepository,
    ProfileNotFound,
    ProfileVersionConflict,
)
from agent.memory.service import MemoryService

router = APIRouter(tags=["记忆"])

# delete-all 二次确认 token（进程内短期存储；多进程部署需换共享存储）
_DELETE_ALL_TTL = timedelta(minutes=5)
_delete_all_tokens: dict[str, dict] = {}


def _service() -> MemoryService | None:
    """从 lifespan 初始化的连接池构建 Service；未初始化返回 None（503）。"""
    factory = memory_database.session_factory
    if factory is None:
        return None
    return MemoryService(MySQLMemoryRepository(factory), MemoryPolicy())


def _require_service() -> MemoryService:
    service = _service()
    if service is None:
        raise HTTPException(status_code=503, detail="Memory v2 未启用或 MySQL 暂不可用")
    return service


def _require_read_service() -> MemoryService:
    """读端点门控（review #8）：READ_ENABLED=0 时不参与回答（21.2 预热状态语义）。"""
    if not memory_database.can_read:
        raise HTTPException(status_code=503, detail="Memory v2 读取未启用（MEMORY_V2_READ_ENABLED=0）")
    return _require_service()


def _require_write_service() -> MemoryService:
    """写端点门控（review #8）：WRITE_ENABLED=0 时拒绝显式写入。"""
    if not memory_database.can_write:
        raise HTTPException(status_code=503, detail="Memory v2 写入未启用（MEMORY_V2_WRITE_ENABLED=0）")
    return _require_service()


# ============================================================
# 请求模型
# ============================================================

class UserScope(BaseModel):
    """user_id 是功能性 scope（方案 8.1）：非空、长度 <=255 的普通字符串。"""

    user_id: str = Field(min_length=1, max_length=255)


class ProfilePatchRequest(UserScope):
    patch: ProfilePatch
    expected_version: int | None = Field(default=None, ge=1)


class CorrectMemoryRequest(UserScope):
    content: str = Field(min_length=1, max_length=2000)
    data: dict | None = None


class ForgetRequest(UserScope):
    reason: str | None = Field(default=None, max_length=500)


class DeleteAllConfirmRequest(BaseModel):
    token: str = Field(min_length=8, max_length=64)


# ============================================================
# Profile
# ============================================================

@router.get("/memory/profile")
async def get_profile(user_id: str = Query(min_length=1, max_length=255)):
    """读取当前用户 Profile（不存在时自动创建空 Profile）。"""
    service = _require_read_service()
    profile = await service.get_profile(user_id)
    return profile.model_dump(mode="json")


@router.patch("/memory/profile")
async def patch_profile(request: Request, body: ProfilePatchRequest):
    """乐观锁更新 Profile（方案 20.2：428 缺失版本 / 409 冲突）。"""
    service = _require_write_service()
    # If-Match header 优先，其次请求体 expected_version；均缺失 → 428
    if_match = request.headers.get("If-Match")
    if if_match is not None:
        try:
            expected_version = int(if_match)
        except ValueError:
            raise HTTPException(status_code=400, detail="If-Match 必须是整数版本号")
    else:
        expected_version = body.expected_version
    if expected_version is None:
        raise HTTPException(status_code=428, detail="缺少 Profile 版本（If-Match 或 expected_version）")

    try:
        profile = await service.update_preference(
            body.user_id,
            body.patch,
            ActorContext(actor_type="user"),
            expected_version=expected_version,
        )
    except ProfileVersionConflict:
        raise HTTPException(status_code=409, detail="Profile 版本冲突，请重新读取后重试")
    except ProfileNotFound:
        # PATCH 不存在的 Profile → 404（方案 20.2：该 scope 内不存在）
        raise HTTPException(status_code=404, detail="Profile 不存在")
    except InvalidProfilePatch:
        # 最终 Profile 状态不合法（review 3.2）→ 400
        raise HTTPException(status_code=400, detail="Profile 变更后状态不合法")
    return profile.model_dump(mode="json")


# ============================================================
# Memory Items
# ============================================================

@router.get("/memories")
async def list_memories(
    user_id: str = Query(min_length=1, max_length=255),
    kind: MemoryKind | None = None,
    status: MemoryStatus | None = None,
    cursor: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
):
    """列表（默认只显示 active；(updated_at, memory_id) opaque cursor 分页）。"""
    service = _require_read_service()
    try:
        page = await service.list_memories(
            user_id,
            MemoryListFilter(kind=kind, status=status),
            cursor=cursor,
            limit=limit,
        )
    except InvalidMemoryCursor as exc:
        # review L：非法 cursor → 400（而非 500）
        raise HTTPException(status_code=400, detail=str(exc))
    return {
        "items": [item.model_dump(mode="json") for item in page.items],
        "next_cursor": page.next_cursor,
        "has_more": page.has_more,
    }


@router.patch("/memories/{memory_id}")
async def correct_memory(memory_id: str, body: CorrectMemoryRequest):
    """纠正记忆：内部创建新 Item 并 supersede 旧 Item（方案 20.2）。"""
    service = _require_write_service()
    try:
        result = await service.correct_memory(
            body.user_id,
            memory_id,
            ActorContext(actor_type="user"),
            content=body.content,
            data=body.data,
        )
    except MemoryNotFound:
        raise HTTPException(status_code=404, detail="记忆不存在")
    except SensitiveContentError as exc:
        raise HTTPException(status_code=400, detail=f"内容命中敏感信息禁存规则（{exc.reason_code}）")
    except InvalidMemoryData:
        # data 不符合 kind 契约（review 3.2）→ 400
        raise HTTPException(status_code=400, detail="data 不符合该记忆类型的契约")
    except InvalidMemoryState:
        # 纠正已 superseded/forgotten 的记忆（review 3.2）→ 409
        raise HTTPException(status_code=409, detail="该记忆不是 active 状态，无法纠正")
    return {"outcome": result.outcome, "memory_id": result.memory_id}


@router.delete("/memories/{memory_id}")
async def forget_memory(memory_id: str, body: ForgetRequest):
    """标记 forgotten：事务提交后立即停止召回（方案 20.4）。"""
    service = _require_write_service()
    result = await service.forget_memory(
        body.user_id,
        ActorContext(actor_type="user"),
        memory_id=memory_id,
        reason=body.reason,
    )
    if result.outcome == "not_found":
        raise HTTPException(status_code=404, detail="记忆不存在")
    return {"deleted": True}


# ============================================================
# delete-all（二次确认，方案 20.1/20.4）
# ============================================================

@router.post("/memories/delete-all/prepare")
async def delete_all_prepare(body: UserScope):
    """返回短期确认 token 和删除范围（不执行删除）。"""
    service = _require_write_service()
    token = uuid.uuid4().hex
    _delete_all_tokens[token] = {
        "user_id": body.user_id,
        "expires_at": datetime.now(timezone.utc) + _DELETE_ALL_TTL,
    }
    return {
        "token": token,
        "expires_at": (datetime.now(timezone.utc) + _DELETE_ALL_TTL).isoformat(),
        "scope": "仅删除长期记忆（memory_profiles/memory_items/记忆相关 job）；"
                 "MongoDB 对话历史属于独立会话数据，不在本次删除范围",
    }


@router.post("/memories/delete-all/confirm")
async def delete_all_confirm(body: DeleteAllConfirmRequest):
    """执行确认后的删除（token 有效期内有效，一次有效）。"""
    service = _require_write_service()
    entry = _delete_all_tokens.pop(body.token, None)
    if entry is None or entry["expires_at"] < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="确认 token 无效或已过期，请重新发起")
    await service.delete_all_memories(entry["user_id"], ActorContext(actor_type="user"))
    return {"deleted": True, "user_id": entry["user_id"]}
