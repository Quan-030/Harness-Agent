# src/agent/memory/repository.py
"""Memory v2 持久化层（方案 5.3 / 17.3 / 18.1）。

- MemoryRepository: 协议（调用方只依赖它，不直接依赖 SQLAlchemy/MySQL driver）
- MySQLMemoryRepository: SQL 查询和持久化，不做 LLM 判断
- 指纹去重（17.3）：命中同 active fingerprint 视为重复，只刷新 updated_at
- 冲突 supersede（17.3）：同 conflict key 内容不同且新事实优先级不低于旧事实时，
  插入新 Item + 旧 Item 标 superseded + 写 event（related_memory_id）
- 所有影响召回结果的写入与 memory_revision += 1 在同一事务（5.5）
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator
from sqlalchemy import and_, bindparam, func, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agent.memory.models import (
    ChartType,
    EntityType,
    MemoryItem,
    MemoryKind,
    MemoryProfile,
    MemoryStatus,
    OutputFormat,
    ProfilePatch,
    SourceType,
    STRICT_MODEL_CONFIG,
    new_uuid7,
    validate_data_by_kind,
    validate_job_payload,
)
from agent.memory.orm_models import (
    MemoryEventORM,
    MemoryItemORM,
    MemoryJobORM,
    MemoryProfileORM,
    MemoryUserStateORM,
)

# ============================================================
# 领域异常
# ============================================================

class ProfileNotFound(Exception):
    """Profile 不存在。"""


class ProfileVersionConflict(Exception):
    """乐观锁冲突：expected_version 与当前 version 不匹配。"""


class MemoryNotFound(Exception):
    """指定 memory_id 不存在。"""


class InvalidMemoryCursor(ValueError):
    """cursor 格式非法（base64/datetime/形状解析失败，review L → API 400）。"""


class InvalidProfilePatch(ValueError):
    """Profile patch 应用后的最终状态不合法（review 3.2 → API 400）。"""


class InvalidMemoryData(ValueError):
    """Memory Item data 不合法（review 3.2 → API 400）。"""


class InvalidMemoryState(ValueError):
    """记忆状态不允许该操作（review 3.2 → API 409）。"""


# ============================================================
# 命令 / 结果类型
# ============================================================

class ActorContext(BaseModel):
    """写入操作的操作者上下文（方案 20.3/5.3：只记录类别，不记录身份）。"""

    model_config = STRICT_MODEL_CONFIG
    actor_type: Literal["user", "system", "admin"]
    source_thread_id: str | None = Field(default=None, max_length=255)


class CreateMemoryCommand(BaseModel):
    """由 Service 构造、Repository 执行的 Memory Item 写入命令。

    memory_id/fingerprint/created_at 等由应用生成，禁止 LLM/API 提供（方案 16.1）。
    supersede_target_id：用户纠正（20.2 PATCH 语义）时显式指定要替代的旧 Item，
    不依赖 conflict_key 推断。
    """

    model_config = STRICT_MODEL_CONFIG
    user_id: str = Field(min_length=1, max_length=255)
    memory_id: str = Field(min_length=36, max_length=36)
    kind: MemoryKind
    content: str = Field(min_length=1, max_length=2000)
    data: dict[str, Any] = Field(default_factory=dict)
    entity_type: EntityType | None = None
    entity_id: str | None = Field(default=None, max_length=255)
    source_type: SourceType
    source_thread_id: str = Field(min_length=1, max_length=255)
    source_message_id: str | None = Field(default=None, max_length=255)
    fingerprint: str = Field(min_length=64, max_length=64)
    expires_at: datetime | None = None
    supersede_target_id: str | None = Field(default=None, max_length=36)

    @model_validator(mode="after")
    def _validate(self) -> "CreateMemoryCommand":
        # 保存 canonical data（review #7）：note_type 等缺省值写入 data，
        # 保证 DB / fingerprint / conflict_key 三处表示一致
        validated = validate_data_by_kind(self.kind, self.data)
        object.__setattr__(self, "data", validated)
        if (self.entity_type is None) != (self.entity_id is None):
            raise ValueError("entity_type 与 entity_id 必须同时为空或同时有值")
        return self


class MemoryWriteResult(BaseModel):
    """create_or_resolve_memory 的结果。

    outcome=conflict 表示低优先级新事实不得覆盖高优先级旧事实，
    本次写入被拒绝（方案 17.3：后台自动路径直接丢弃；显式路径返回冲突）。
    """

    model_config = STRICT_MODEL_CONFIG
    memory_id: str
    outcome: Literal["created", "duplicate", "superseded", "conflict"]
    superseded_memory_ids: list[str] = Field(default_factory=list, max_length=50)


class MemoryListFilter(BaseModel):
    """list_memories 过滤条件（20.2：默认只显示 active）。"""

    model_config = STRICT_MODEL_CONFIG
    kind: MemoryKind | None = None
    status: MemoryStatus | None = None


class MemoryPage(BaseModel):
    """(updated_at, memory_id) opaque cursor 分页（方案 20.2）。"""

    model_config = STRICT_MODEL_CONFIG
    items: list[MemoryItem]
    next_cursor: str | None = None
    has_more: bool = False


class MemoryJob(BaseModel):
    """memory_jobs 域模型（worker 消费）。"""

    model_config = STRICT_MODEL_CONFIG
    job_id: str
    idempotency_key: str
    user_id: str
    thread_id: str
    job_type: Literal["extract_memory", "expire_memory"]
    payload: dict[str, Any]
    status: Literal["pending", "processing", "succeeded", "failed", "dead", "cancelled"]
    attempts: int = Field(ge=0)
    available_at: datetime
    locked_at: datetime | None = None
    locked_by: str | None = None
    last_error: str | None = None
    created_at: datetime
    updated_at: datetime


class EnqueueJobCommand(BaseModel):
    """入队命令（19.1：事务内读 generation 写入 payload 后插入）。"""

    model_config = STRICT_MODEL_CONFIG
    user_id: str = Field(min_length=1, max_length=255)
    thread_id: str = Field(min_length=1, max_length=255)
    job_type: Literal["extract_memory", "expire_memory"]
    payload: dict[str, Any]


# ============================================================
# 幂等键（方案 19.2：memory_jobs.idempotency_key 的唯一权威定义）
# ============================================================

def compute_idempotency_key(
    *,
    job_type: str,
    user_id: str,
    thread_id: str,
    assistant_message_id: str | None = None,
    extractor_version: str | None = None,
    memory_generation: int,
    replay_generation: int,
    scan_partition: str | None = None,
    cutoff_at: str | None = None,
    policy_version: str | None = None,
) -> str:
    """canonical JSON（UTF-8、key 排序、无多余空格）SHA-256。

    checkpoint_id 不参与 extract_memory 幂等键（19.2：同一最终回复在恢复/重试时
    可能关联不同 checkpoint，不能因此创建重复抽取任务）。
    """
    if job_type == "extract_memory":
        fields = {
            "job_type": job_type,
            "user_id": user_id,
            "thread_id": thread_id,
            "assistant_message_id": assistant_message_id,
            "extractor_version": extractor_version,
            "memory_generation": memory_generation,
            "replay_generation": replay_generation,
        }
    else:  # expire_memory
        fields = {
            "job_type": job_type,
            "user_id": user_id,
            "scan_partition": scan_partition,
            "cutoff_at": cutoff_at,
            "policy_version": policy_version,
            "memory_generation": memory_generation,
            "replay_generation": replay_generation,
        }
    canonical = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ============================================================
# 协议（方案 5.1：调用方只依赖 MemoryRepository）
# ============================================================

class MemoryRepository(Protocol):
    async def get_or_create_profile(self, user_id: str) -> MemoryProfile: ...
    async def patch_profile(
        self, user_id: str, patch: ProfilePatch, expected_version: int,
        actor: ActorContext,
    ) -> MemoryProfile: ...
    async def create_or_resolve_memory(
        self, command: CreateMemoryCommand, actor: ActorContext,
    ) -> MemoryWriteResult: ...
    async def forget_memory(
        self, user_id: str, memory_id: str, reason: str | None, actor: ActorContext,
    ) -> None: ...
    async def get_memory(self, user_id: str, memory_id: str) -> MemoryItem: ...
    async def delete_all_memories(self, user_id: str, actor: ActorContext) -> None: ...
    async def search_memories(
        self, query: "RecallQuery",
    ) -> list[tuple[MemoryItem, float]]: ...
    async def enqueue_job(self, command: EnqueueJobCommand) -> str: ...
    async def claim_jobs(
        self, worker_id: str, limit: int, now: datetime,
    ) -> list[MemoryJob]: ...
    async def complete_job(self, job_id: str, worker_id: str) -> None: ...
    async def fail_job(
        self, job_id: str, worker_id: str, error: str, retry_at: datetime,
    ) -> None: ...
    async def get_job_by_idempotency_key(self, key: str) -> MemoryJob | None: ...
    async def get_memory_generation(self, user_id: str) -> int: ...
    async def apply_extract_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        user_id: str,
        expected_generation: int,
        commands: list["CreateMemoryCommand"],
        actor: ActorContext,
    ) -> Literal["succeeded", "cancelled", "stale"]: ...
    async def cancel_job(self, job_id: str, worker_id: str) -> None: ...
    async def dead_job(self, job_id: str, worker_id: str, error: str) -> None: ...
    async def apply_expire_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        user_id: str,
        expected_generation: int,
        cutoff: str,
    ) -> Literal["succeeded", "cancelled", "stale"]: ...
    async def recover_due_jobs(
        self, now: datetime, *, lease_seconds: int, max_attempts: int,
    ) -> int: ...
    async def list_memories(
        self, user_id: str, filters: MemoryListFilter,
        cursor: str | None, limit: int,
    ) -> MemoryPage: ...
    async def get_memory_revision(self, user_id: str) -> int: ...
    async def find_memories_for_forget(
        self, user_id: str, query: str, limit: int = 21,
    ) -> list[MemoryItem]: ...


# ============================================================
# MySQL 实现
# ============================================================

# 来源可靠性优先级（方案 6.4：用户明确陈述 > 工具验证 > 模型推断）
_SOURCE_PRIORITY: dict[SourceType, int] = {
    SourceType.USER_EXPLICIT: 3,
    SourceType.TOOL_VERIFIED: 2,
    SourceType.MODEL_INFERRED: 1,
}


def _encode_cursor(updated_at: datetime, memory_id: str) -> str:
    raw = f"{updated_at.isoformat()}|{memory_id}"
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def _decode_cursor(cursor: str) -> tuple[datetime, str]:
    """解析 opaque cursor（review L：非法格式抛 InvalidMemoryCursor → API 400）。"""
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8")
        updated_at_str, memory_id = raw.rsplit("|", 1)
        updated_at = datetime.fromisoformat(updated_at_str)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidMemoryCursor(f"非法 cursor: {exc}") from exc
    return updated_at, memory_id


def _job_to_domain(row: MemoryJobORM) -> MemoryJob:
    """ORM job 行 → 域模型。"""
    return MemoryJob(
        job_id=row.job_id,
        idempotency_key=row.idempotency_key,
        user_id=row.user_id,
        thread_id=row.thread_id,
        job_type=row.job_type,
        payload=row.payload,
        status=row.status,
        attempts=row.attempts,
        available_at=_utc(row.available_at),
        locked_at=None if row.locked_at is None else _utc(row.locked_at),
        locked_by=row.locked_by,
        last_error=row.last_error,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )


def _row_to_item(row) -> MemoryItem:
    """raw SQL 行（16 列：15 个字段 + ft_relevance）→ Pydantic 领域模型。"""
    data = row[4]
    if isinstance(data, str):  # text() 无类型映射：JSON 列返回字符串，需解析
        data = json.loads(data)
    return MemoryItem(
        memory_id=row[0],
        user_id=row[1],
        kind=MemoryKind(row[2]),
        content=row[3],
        data=data,
        entity_type=None if row[5] is None else EntityType(row[5]),
        entity_id=row[6],
        source_type=SourceType(row[7]),
        source_thread_id=row[8],
        source_message_id=row[9],
        status=MemoryStatus(row[10]),
        fingerprint=row[11],
        created_at=_utc(row[12]),
        updated_at=_utc(row[13]),
        expires_at=None if row[14] is None else _utc(row[14]),
    )


def _to_domain_item(row: MemoryItemORM) -> MemoryItem:
    """ORM 行 → Pydantic 领域模型（DATETIME 转 UTC-aware，方案 16.1）。"""
    return MemoryItem(
        memory_id=row.memory_id,
        user_id=row.user_id,
        kind=MemoryKind(row.kind),
        content=row.content,
        data=row.data,
        entity_type=None if row.entity_type is None else EntityType(row.entity_type),
        entity_id=row.entity_id,
        source_type=SourceType(row.source_type),
        source_thread_id=row.source_thread_id,
        source_message_id=row.source_message_id,
        status=MemoryStatus(row.status),
        fingerprint=row.fingerprint,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
        expires_at=None if row.expires_at is None else _utc(row.expires_at),
    )


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class MySQLMemoryRepository:
    """SQL 查询和持久化，不做 LLM 判断（方案 18.1 分层职责）。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]):
        self._session_factory = session_factory

    # ---------- 内部工具 ----------

    async def _ensure_user_state(self, session: AsyncSession, user_id: str) -> None:
        """race-safe 原子 upsert memory_user_state（方案 5.5，review #3）。

        并发首次访问不会产生 PK 冲突（ON DUPLICATE KEY 为 no-op）。
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        await session.execute(
            text(
                "INSERT INTO memory_user_state "
                "(user_id, memory_revision, memory_generation, created_at, updated_at) "
                "VALUES (:user_id, 0, 0, :now, :now) "
                "ON DUPLICATE KEY UPDATE user_id = user_id"
            ),
            {"user_id": user_id, "now": now},
        )

    async def _lock_user_state(self, session: AsyncSession, user_id: str) -> None:
        """per-user serialization lock（review C）：原子 ensure 后 SELECT ... FOR UPDATE。

        所有 generation / recall-state mutation（create/forget/delete-all/未来 worker）
        统一在此锁内执行，形成一致的事务锁协议，避免各路径不同锁顺序。
        """
        await self._ensure_user_state(session, user_id)
        await session.execute(
            select(MemoryUserStateORM.user_id)
            .where(MemoryUserStateORM.user_id == user_id)
            .with_for_update()
        )

    async def _bump_revision(self, session: AsyncSession, user_id: str) -> None:
        """同事务内 memory_revision += 1（方案 5.5：任何影响召回结果的写入）。"""
        await session.execute(
            text(
                "UPDATE memory_user_state SET memory_revision = memory_revision + 1, "
                "updated_at = :now WHERE user_id = :user_id"
            ),
            {"user_id": user_id, "now": datetime.now(timezone.utc).replace(tzinfo=None)},
        )

    async def _append_event(
        self,
        session: AsyncSession,
        *,
        user_id: str,
        memory_id: str | None,
        event_type: str,
        actor: ActorContext,
        related_memory_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        from agent.memory.models import new_uuid7
        session.add(
            MemoryEventORM(
                event_id=new_uuid7(),
                user_id=user_id,
                memory_id=memory_id,
                event_type=event_type,
                related_memory_id=related_memory_id,
                source_thread_id=actor.source_thread_id,
                actor_type=actor.actor_type,
                reason=reason,
                created_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
        )

    # ---------- Profile ----------

    async def get_or_create_profile(self, user_id: str) -> MemoryProfile:
        async with self._session_factory() as session:
            async with session.begin():
                await self._lock_user_state(session, user_id)
                row = await session.get(MemoryProfileORM, user_id)
                if row is None:
                    # 原子 upsert 消除并发首次访问的 PK 冲突（review #3）
                    now = datetime.now(timezone.utc).replace(tzinfo=None)
                    await session.execute(
                        text(
                            "INSERT INTO memory_profiles "
                            "(user_id, procurement_defaults, created_at, updated_at) "
                            "VALUES (:user_id, JSON_OBJECT(), :now, :now) "
                            "ON DUPLICATE KEY UPDATE user_id = user_id"
                        ),
                        {"user_id": user_id, "now": now},
                    )
                    row = await session.get(MemoryProfileORM, user_id)
                return _profile_to_domain(row)

    async def patch_profile(
        self, user_id: str, patch: ProfilePatch, expected_version: int,
        actor: ActorContext,
    ) -> MemoryProfile:
        async with self._session_factory() as session:
            async with session.begin():
                await self._lock_user_state(session, user_id)
                row = await session.get(MemoryProfileORM, user_id)
                if row is None:
                    raise ProfileNotFound(f"Profile 不存在: {user_id}")
                if row.version != expected_version:
                    raise ProfileVersionConflict(
                        f"Profile 版本冲突：期望 {expected_version}，当前 {row.version}"
                    )

                # 真正的 CAS（review A/N）：
                # - 不在 CAS 成功前 mutate attached ORM row（autoflush 会绕过 version 条件）
                # - 纯内存计算 patch 结果 → 完整 MemoryProfile Pydantic 校验（最终状态契约）
                # - 单条条件 UPDATE ... WHERE version = :expected 原子执行
                candidate_values = _apply_patch_to_values(
                    _profile_row_to_values(row), patch
                )
                candidate_values["user_id"] = user_id
                candidate_values["version"] = expected_version + 1
                candidate_values["created_at"] = _utc(row.created_at)
                candidate_values["updated_at"] = datetime.now(timezone.utc)
                # 最终状态完整校验（review N：49 条 + add 2 = 51 条在此拒绝）。
                # strict 模式下枚举需先转回实例（DB 存的是字符串）。
                # 客户端错误（review 3.2）→ InvalidProfilePatch，不泄漏 ValidationError
                candidate = dict(candidate_values)
                if candidate.get("output_format") is not None:
                    candidate["output_format"] = OutputFormat(candidate["output_format"])
                if candidate.get("chart_type") is not None:
                    candidate["chart_type"] = ChartType(candidate["chart_type"])
                try:
                    MemoryProfile.model_validate(candidate)
                except Exception as exc:
                    raise InvalidProfilePatch(f"Profile 最终状态不合法: {exc}") from exc

                now = candidate_values["updated_at"].replace(tzinfo=None)
                result = await session.execute(
                    text(
                        "UPDATE memory_profiles SET "
                        "version = version + 1, "
                        "output_format = :output_format, "
                        "chart_type = :chart_type, "
                        "currency = :currency, "
                        "language = :language, "
                        "procurement_defaults = :procurement_defaults, "
                        "updated_at = :now "
                        "WHERE user_id = :user_id AND version = :expected_version"
                    ),
                    {
                        "user_id": user_id,
                        "expected_version": expected_version,
                        "output_format": _enum_value(candidate_values.get("output_format")),
                        "chart_type": _enum_value(candidate_values.get("chart_type")),
                        "currency": candidate_values.get("currency"),
                        "language": candidate_values.get("language"),
                        "procurement_defaults": json.dumps(
                            candidate_values["procurement_defaults"]
                        ),
                        "now": now,
                    },
                )
                if result.rowcount == 0:
                    raise ProfileVersionConflict(
                        f"Profile 版本冲突：期望 {expected_version}"
                    )

                await self._append_event(
                    session,
                    user_id=user_id,
                    memory_id=None,
                    event_type="updated",
                    actor=actor,
                    reason="profile patch",
                )
                await self._bump_revision(session, user_id)
                # asyncmy 无 INSERT/UPDATE RETURNING，且 raw UPDATE 不会失效
                # identity map 缓存：显式 refresh 从 DB 回读最新行
                await session.refresh(row)
                return _profile_to_domain(row)

    # ---------- Memory Item ----------

    async def create_or_resolve_memory(
        self, command: CreateMemoryCommand, actor: ActorContext,
    ) -> MemoryWriteResult:
        """写入/去重/冲突处理。per-user mutex 内串行执行（review #2/C）。

        IntegrityError（UNIQUE 最终防线触发）→ 整个事务已失败，开启新事务重查
        active fingerprint（review D）；per-user 锁协议下正常路径不应触发。
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        try:
            async with self._session_factory() as session:
                async with session.begin():
                    await self._lock_user_state(session, command.user_id)
                    result = await self._create_or_resolve_in_session(
                        session, command, actor, now, bump_revision=True
                    )
                return result
        except IntegrityError:
            async with self._session_factory() as session:
                async with session.begin():
                    existing = await session.execute(
                        select(MemoryItemORM).where(
                            MemoryItemORM.user_id == command.user_id,
                            MemoryItemORM.status == "active",
                            MemoryItemORM.fingerprint == command.fingerprint,
                        )
                    )
                    dup = existing.scalar_one_or_none()
            if dup is not None:
                return MemoryWriteResult(
                    memory_id=dup.memory_id,
                    outcome="duplicate",
                    superseded_memory_ids=[],
                )
            raise

    async def _create_or_resolve_in_session(
        self,
        session: AsyncSession,
        command: CreateMemoryCommand,
        actor: ActorContext,
        now: datetime,
        *,
        bump_revision: bool,
    ) -> MemoryWriteResult:
        """session-aware 写入核心（review #1.2：不开事务、不拿 user 锁，由调用方负责）。

        供 create_or_resolve_memory 与 worker 的 apply_extract_job 在同一事务内复用；
        conflict blocked 路径不 rollback（前置判定已保证未修改任何 row），
        避免单个候选被拒时回滚同一 worker 事务内已处理的候选。
        """
        # 1. 指纹去重（方案 17.3）：同 active fingerprint 视为重复
        existing = await session.execute(
            select(MemoryItemORM).where(
                MemoryItemORM.user_id == command.user_id,
                MemoryItemORM.status == "active",
                MemoryItemORM.fingerprint == command.fingerprint,
            )
        )
        dup = existing.scalar_one_or_none()
        if dup is not None:
            # 重复：只刷新 updated_at，可补齐缺失的 source message；不得降低来源可靠性。
            # updated_at 参与 recency/排序/cursor → 影响召回结果（review J）
            dup.updated_at = now
            if command.source_message_id and not dup.source_message_id:
                dup.source_message_id = command.source_message_id
            if bump_revision:
                await self._bump_revision(session, command.user_id)
            await session.flush()
            return MemoryWriteResult(
                memory_id=dup.memory_id,
                outcome="duplicate",
                superseded_memory_ids=[],
            )

        # 2. 冲突检测（方案 17.3 conflict_key；blocked 前置判定，不修改 row）
        superseded_ids = []
        if command.supersede_target_id is not None:
            # 用户显式纠正（20.2 PATCH）：替代指定 Item，不依赖 conflict_key
            superseded_ids = await self._supersede_target(
                session, command, now, actor
            )
        else:
            superseded_ids, blocked = await self._resolve_conflicts(
                session, command, now, actor
            )
            if blocked:
                # 低优先级信息不得自动覆盖高优先级信息：本次写入被拒绝（方案 17.3）。
                # 不 rollback：前置判定未修改任何 row，调用方可继续处理其他候选
                return MemoryWriteResult(
                    memory_id=command.memory_id,
                    outcome="conflict",
                    superseded_memory_ids=[],
                )

        # 3. 写入新 Item + event
        session.add(
            MemoryItemORM(
                memory_id=command.memory_id,
                user_id=command.user_id,
                kind=command.kind.value,
                content=command.content,
                data=command.data,
                entity_type=None if command.entity_type is None else command.entity_type.value,
                entity_id=command.entity_id,
                source_type=command.source_type.value,
                source_thread_id=command.source_thread_id,
                source_message_id=command.source_message_id,
                status="active",
                fingerprint=command.fingerprint,
                created_at=now,
                updated_at=now,
                expires_at=None if command.expires_at is None else command.expires_at.replace(tzinfo=None),
            )
        )
        await self._append_event(
            session,
            user_id=command.user_id,
            memory_id=command.memory_id,
            event_type="created",
            actor=actor,
        )
        if bump_revision:
            await self._bump_revision(session, command.user_id)
        await session.flush()
        return MemoryWriteResult(
            memory_id=command.memory_id,
            outcome="created" if not superseded_ids else "superseded",
            superseded_memory_ids=superseded_ids,
        )

    async def _resolve_conflicts(
        self,
        session: AsyncSession,
        command: CreateMemoryCommand,
        now: datetime,
        actor: ActorContext,
    ) -> tuple[list[str], bool]:
        """按 conflict_key 查找同 scope 的 active 旧事实并 supersede。

        task_outcome / user_feedback 默认不冲突（方案 17.3）。
        返回 (superseded_ids, blocked)。

        review #1.2：**两阶段实现**——先收集全部同 conflict_key 冲突并判定是否存在
        更高优先级旧事实；存在则返回 blocked 且**不修改任何 row**（外层不需要
        rollback，避免把同一 worker 事务内已处理的候选一并回滚）；确认允许覆盖后
        才执行 supersede。
        """
        if command.kind in (MemoryKind.TASK_OUTCOME, MemoryKind.USER_FEEDBACK):
            return [], False

        # 候选：同 kind、同实体（entity 为空时匹配实体为空或任意）
        stmt = select(MemoryItemORM).where(
            MemoryItemORM.user_id == command.user_id,
            MemoryItemORM.status == "active",
            MemoryItemORM.kind == command.kind.value,
        )
        rows = (await session.execute(stmt)).scalars().all()

        conflict_key_field = {
            MemoryKind.SUPPLIER_CONTEXT: "note_type",
            MemoryKind.PROCUREMENT_CONSTRAINT: "constraint_name",
        }[command.kind]

        # 阶段 1：收集冲突目标并判定 blocked（不修改任何 row）
        targets: list[MemoryItemORM] = []
        for old in rows:
            if old.entity_type != (
                None if command.entity_type is None else command.entity_type.value
            ) or old.entity_id != command.entity_id:
                continue
            old_key = _conflict_key(old, conflict_key_field)
            new_key = command.data.get(conflict_key_field)
            if old_key != new_key:
                continue
            # 内容相同由指纹去重兜底；这里只处理内容不同的冲突
            if old.content == command.content and old.data == command.data:
                continue
            # 优先级比较：低优先级不得自动覆盖高优先级（方案 17.3）
            if _SOURCE_PRIORITY[command.source_type] < _SOURCE_PRIORITY[SourceType(old.source_type)]:
                return [], True
            targets.append(old)

        # 阶段 2：确认允许覆盖后统一执行 supersede（此时无 blocked）
        superseded_ids: list[str] = []
        for old in targets:
            old.status = "superseded"
            old.updated_at = now
            await self._append_event(
                session,
                user_id=command.user_id,
                memory_id=old.memory_id,
                event_type="superseded",
                actor=actor,
                related_memory_id=command.memory_id,
                reason="conflict supersede",
            )
            superseded_ids.append(old.memory_id)
        return superseded_ids, False

    async def _supersede_target(
        self,
        session: AsyncSession,
        command: CreateMemoryCommand,
        now: datetime,
        actor: ActorContext,
    ) -> list[str]:
        """用户显式纠正：将指定旧 Item 标 superseded（方案 20.2 PATCH 语义）。"""
        assert command.supersede_target_id is not None
        old = await session.get(MemoryItemORM, command.supersede_target_id)
        if old is None or old.user_id != command.user_id:
            raise MemoryNotFound(f"记忆不存在: {command.supersede_target_id}")
        if old.status != "active":
            # review 3.2：纠正已 superseded/forgotten 记忆 → InvalidMemoryState（API 409）
            raise InvalidMemoryState(
                f"记忆 {command.supersede_target_id} 不是 active（当前 {old.status}），无法纠正"
            )
        old.status = "superseded"
        old.updated_at = now
        await self._append_event(
            session,
            user_id=command.user_id,
            memory_id=old.memory_id,
            event_type="superseded",
            actor=actor,
            related_memory_id=command.memory_id,
            reason="user correction",
        )
        return [old.memory_id]

    async def forget_memory(
        self, user_id: str, memory_id: str, reason: str | None, actor: ActorContext,
    ) -> None:
        """普通 forget：同事务内标 forgotten + 写 event + revision += 1（方案 20.4）。"""
        async with self._session_factory() as session:
            async with session.begin():
                await self._lock_user_state(session, user_id)
                row = await session.get(MemoryItemORM, memory_id)
                if row is None or row.user_id != user_id:
                    raise MemoryNotFound(f"记忆不存在: {memory_id}")
                row.status = "forgotten"
                row.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
                await self._append_event(
                    session,
                    user_id=user_id,
                    memory_id=memory_id,
                    event_type="forgotten",
                    actor=actor,
                    reason=reason,
                )
                await self._bump_revision(session, user_id)
                await session.flush()

    async def get_memory(self, user_id: str, memory_id: str) -> MemoryItem:
        """按 ID 读取单条记忆（scope 校验：其他用户的记忆视为不存在）。"""
        async with self._session_factory() as session:
            row = await session.get(MemoryItemORM, memory_id)
            if row is None or row.user_id != user_id:
                raise MemoryNotFound(f"记忆不存在: {memory_id}")
            return _to_domain_item(row)

    async def delete_all_memories(self, user_id: str, actor: ActorContext) -> None:
        """delete-all（方案 20.4）：一个事务内完成全部长期记忆的停止召回。

        - memory_generation += 1 与 memory_revision += 1（防止旧 job 复活记忆）
        - 删除 Profile；active/superseded Items 全部标记 forgotten
        - pending/failed Job 标记 cancelled；processing Job 留给 worker 按 generation 检查
        - 审计事件（memory_id=NULL）
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._session_factory() as session:
            async with session.begin():
                await self._lock_user_state(session, user_id)
                # generation += 1（阻止入队于 delete-all 前的 job 复活记忆，方案 19.1）
                await session.execute(
                    text(
                        "UPDATE memory_user_state SET "
                        "memory_generation = memory_generation + 1, "
                        "memory_revision = memory_revision + 1, updated_at = :now "
                        "WHERE user_id = :user_id"
                    ),
                    {"user_id": user_id, "now": now},
                )
                # 删除 Profile（该用户从零开始）
                row = await session.get(MemoryProfileORM, user_id)
                if row is not None:
                    await session.delete(row)
                # 停止所有可召回 Item 的召回
                await session.execute(
                    text(
                        "UPDATE memory_items SET status='forgotten', updated_at=:now "
                        "WHERE user_id=:user_id AND status IN ('active','superseded')"
                    ),
                    {"user_id": user_id, "now": now},
                )
                # 取消未执行/失败 job（processing 由 worker 按 generation 检查）
                await session.execute(
                    text(
                        "UPDATE memory_jobs SET status='cancelled', updated_at=:now "
                        "WHERE user_id=:user_id AND status IN ('pending','failed')"
                    ),
                    {"user_id": user_id, "now": now},
                )
                await self._append_event(
                    session,
                    user_id=user_id,
                    memory_id=None,
                    event_type="forgotten",
                    actor=actor,
                    reason="delete-all",
                )
                await session.flush()

    # ---------- Job queue（方案 19） ----------

    async def enqueue_job(self, command: EnqueueJobCommand) -> str:
        """入队协议（19.1）：单事务内 get-or-create user_state → SELECT generation
        FOR UPDATE → generation 写入 payload → 计算幂等键 → 插入 job。

        幂等键 UNIQUE 冲突（重试/恢复重复入队）→ 返回既有 job_id（幂等，19.2）。
        """
        # payload 按 job_type 校验（extra=forbid）
        payload = validate_job_payload(command.job_type, command.payload)
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_user_state(session, command.user_id)
                # 事务内读 generation（19.1：不得在事务外读取后再入队）
                row = (
                    await session.execute(
                        text(
                            "SELECT memory_generation FROM memory_user_state "
                            "WHERE user_id = :user_id FOR UPDATE"
                        ),
                        {"user_id": command.user_id},
                    )
                ).first()
                generation = int(row[0]) if row else 0
                payload["memory_generation"] = generation

                job_id = new_uuid7()
                idempotency_key = compute_idempotency_key(
                    job_type=command.job_type,
                    user_id=command.user_id,
                    thread_id=command.thread_id,
                    assistant_message_id=payload.get("assistant_message_id"),
                    extractor_version=payload.get("extractor_version"),
                    memory_generation=generation,
                    replay_generation=payload.get("replay_generation", 0),
                    scan_partition=payload.get("scan_partition"),
                    cutoff_at=payload.get("cutoff_at"),
                    policy_version=payload.get("policy_version"),
                )
                session.add(
                    MemoryJobORM(
                        job_id=job_id,
                        idempotency_key=idempotency_key,
                        user_id=command.user_id,
                        thread_id=command.thread_id,
                        job_type=command.job_type,
                        payload=payload,
                        status="pending",
                        attempts=0,
                        available_at=now,
                        created_at=now,
                        updated_at=now,
                    )
                )
                try:
                    await session.flush()
                except IntegrityError:
                    # 幂等键冲突：返回既有 job（19.2 重试/中断恢复不重复写）
                    await session.rollback()
                    existing = await self.get_job_by_idempotency_key(idempotency_key)
                    if existing is not None:
                        return existing.job_id
                    raise
                return job_id

    async def claim_jobs(
        self, worker_id: str, limit: int, now: datetime,
    ) -> list[MemoryJob]:
        """领取任务（19.3）：SELECT ... FOR UPDATE SKIP LOCKED，改 processing、
        设置 lock、attempts += 1。多个 worker 不会重复领取同一 job。"""
        rows: list[MemoryJob] = []
        async with self._session_factory() as session:
            async with session.begin():
                stmt = (
                    select(MemoryJobORM)
                    .where(
                        MemoryJobORM.status == "pending",
                        MemoryJobORM.available_at <= now.replace(tzinfo=None),
                    )
                    .order_by(MemoryJobORM.available_at)
                    .limit(limit)
                    .with_for_update(skip_locked=True)
                )
                jobs = (await session.execute(stmt)).scalars().all()
                for job in jobs:
                    job.status = "processing"
                    job.locked_at = now.replace(tzinfo=None)
                    job.locked_by = worker_id
                    job.attempts += 1
                    job.updated_at = now.replace(tzinfo=None)
                    rows.append(_job_to_domain(job))
                await session.flush()
        return rows

    async def complete_job(self, job_id: str, worker_id: str) -> None:
        """标记 succeeded 并清空 lock（19.3：终态清锁）。"""
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(MemoryJobORM, job_id)
                if job is None or job.locked_by != worker_id:
                    return  # 锁不属于该 worker：lease 已接管，忽略
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                job.status = "succeeded"
                job.locked_at = None
                job.locked_by = None
                job.updated_at = now
                await session.flush()

    async def fail_job(
        self, job_id: str, worker_id: str, error: str, retry_at: datetime,
    ) -> None:
        """标记 failed：清锁、设置 available_at（退避后重试）、记录安全错误摘要。

        last_error 不保存消息正文、凭据或 DSN（方案 19.3 / 20.3）。
        """
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(MemoryJobORM, job_id)
                if job is None or job.locked_by != worker_id:
                    return
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                job.status = "failed"
                job.locked_at = None
                job.locked_by = None
                job.available_at = retry_at.replace(tzinfo=None)
                job.last_error = error[:2000]
                job.updated_at = now
                await session.flush()

    async def apply_extract_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        user_id: str,
        expected_generation: int,
        commands: list[CreateMemoryCommand],
        actor: ActorContext,
    ) -> Literal["succeeded", "cancelled", "stale"]:
        """worker 固定事务协议（方案 19.3 + review #1）：同一事务内完成
        generation fence + lease ownership 校验、候选写入、revision 递增、job 终态。

        - stale：job 已非 processing 或锁不属于该 worker（lease 被 reaper 回收后
          旧 worker 不得写入）——不写任何记忆
        - cancelled：generation 不匹配（delete-all 后旧 job 不得复活记忆）
        - succeeded：校验通过，候选全部写入后 job 成功

        候选写入复用 _create_or_resolve_in_session（同一 session），
        conflict blocked 只拒绝该候选、不回滚其他候选。
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_user_state(session, user_id)

                # 1. generation fence：锁 user_state 后读取（防 delete-all 并发）
                state_row = (
                    await session.execute(
                        text(
                            "SELECT memory_generation FROM memory_user_state "
                            "WHERE user_id = :user_id FOR UPDATE"
                        ),
                        {"user_id": user_id},
                    )
                ).first()
                current_generation = int(state_row[0]) if state_row else 0

                # 2. lease ownership：job 行 FOR UPDATE，校验 processing + 锁归属
                job = (
                    await session.execute(
                        select(MemoryJobORM)
                        .where(MemoryJobORM.job_id == job_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if job is None or job.status != "processing" or job.locked_by != worker_id:
                    return "stale"

                # 3. generation 不匹配 → cancelled（终态，清锁；不允许 replay，19.3）
                if current_generation != expected_generation:
                    job.status = "cancelled"
                    job.locked_at = None
                    job.locked_by = None
                    job.updated_at = now
                    await session.flush()
                    return "cancelled"

                # 4. 校验通过：写入候选（同一 session；blocked 只拒该候选）
                has_change = False
                for command in commands:
                    result = await self._create_or_resolve_in_session(
                        session, command, actor, now, bump_revision=False
                    )
                    if result.outcome != "conflict":
                        has_change = True
                if has_change:
                    # 任一候选产生召回可见变化（创建/去重刷新/替代）→ revision += 1
                    await self._bump_revision(session, user_id)

                # 5. job → succeeded（清锁）
                job.status = "succeeded"
                job.locked_at = None
                job.locked_by = None
                job.updated_at = now
                await session.flush()
                return "succeeded"

    async def apply_expire_job(
        self,
        *,
        job_id: str,
        worker_id: str,
        user_id: str,
        expected_generation: int,
        cutoff: str,
    ) -> Literal["succeeded", "cancelled", "stale"]:
        """expire_memory 的 generation fence 原子事务（review 第二轮）。

        与 apply_extract_job 对称：同一事务内完成 generation fence + lease ownership
        校验、过期 Item 清理、revision 递增、job 终态。旧 generation 的 expire job
        不得跨 delete-all 边界修改新状态。
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_user_state(session, user_id)

                # 1. generation fence：锁 user_state 后读取
                state_row = (
                    await session.execute(
                        text(
                            "SELECT memory_generation FROM memory_user_state "
                            "WHERE user_id = :user_id FOR UPDATE"
                        ),
                        {"user_id": user_id},
                    )
                ).first()
                current_generation = int(state_row[0]) if state_row else 0

                # 2. lease ownership：job 行 FOR UPDATE
                job = (
                    await session.execute(
                        select(MemoryJobORM)
                        .where(MemoryJobORM.job_id == job_id)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if job is None or job.status != "processing" or job.locked_by != worker_id:
                    return "stale"

                # 3. generation 不匹配 → cancelled（清锁）
                if current_generation != expected_generation:
                    job.status = "cancelled"
                    job.locked_at = None
                    job.locked_by = None
                    job.updated_at = now
                    await session.flush()
                    return "cancelled"

                # 4. 清理过期 Item（同事务）
                result = await session.execute(
                    text(
                        "UPDATE memory_items SET status='forgotten', updated_at=:now "
                        "WHERE user_id=:user_id AND status='active' "
                        "AND expires_at IS NOT NULL AND expires_at < :cutoff"
                    ),
                    {
                        "user_id": user_id,
                        "cutoff": cutoff,
                        "now": now,
                    },
                )
                if (result.rowcount or 0) > 0:
                    # 过期清理改变召回结果 → revision += 1
                    await self._bump_revision(session, user_id)

                # 5. job → succeeded（清锁）
                job.status = "succeeded"
                job.locked_at = None
                job.locked_by = None
                job.updated_at = now
                await session.flush()
                return "succeeded"

    async def expire_items_before(self, user_id: str, cutoff: str) -> int:
        """expire_memory：将 expires_at 早于 cutoff 的 Item 停止召回（18.4 宽限期语义）。

        Returns:
            受影响行数。
        """
        async with self._session_factory() as session:
            async with session.begin():
                await self._ensure_user_state(session, user_id)
                result = await session.execute(
                    text(
                        "UPDATE memory_items SET status='forgotten', updated_at=:now "
                        "WHERE user_id=:user_id AND status='active' "
                        "AND expires_at IS NOT NULL AND expires_at < :cutoff"
                    ),
                    {
                        "user_id": user_id,
                        "cutoff": cutoff,
                        "now": datetime.now(timezone.utc).replace(tzinfo=None),
                    },
                )
                await self._bump_revision(session, user_id)
                return result.rowcount or 0

    async def get_memory_generation(self, user_id: str) -> int:
        async with self._session_factory() as session:
            row = await session.get(MemoryUserStateORM, user_id)
            return row.memory_generation if row is not None else 0

    async def cancel_job(self, job_id: str, worker_id: str) -> None:
        """generation 不匹配 → cancelled（终态，清锁；不允许自动重试或 replay，19.3）。"""
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(MemoryJobORM, job_id)
                if job is None or job.locked_by != worker_id:
                    return
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                job.status = "cancelled"
                job.locked_at = None
                job.locked_by = None
                job.updated_at = now
                await session.flush()

    async def dead_job(self, job_id: str, worker_id: str, error: str) -> None:
        """超过最大尝试 → dead（终态，清锁，保留安全错误摘要，19.3）。"""
        async with self._session_factory() as session:
            async with session.begin():
                job = await session.get(MemoryJobORM, job_id)
                if job is None or job.locked_by != worker_id:
                    return
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                job.status = "dead"
                job.locked_at = None
                job.locked_by = None
                job.last_error = error[:2000]
                job.updated_at = now
                await session.flush()

    async def recover_due_jobs(
        self, now: datetime, *, lease_seconds: int, max_attempts: int,
    ) -> int:
        """reaper（方案 19.3 + review #4）：恢复/死信到期 job。

        - processing 且 lease 超时（locked_at <= now - lease）：
          attempts 未耗尽 → pending（清锁）；耗尽 → dead（清锁）
        - failed 且 available_at 到期：attempts 未耗尽 → pending；耗尽 → dead

        lease_seconds 由 worker 实例传入（不硬编码，review #4）。

        Returns:
            转为 pending 的 job 数量。
        """
        now_naive = now.replace(tzinfo=None)
        lease_cutoff = (now - timedelta(seconds=lease_seconds)).replace(tzinfo=None)
        async with self._session_factory() as session:
            async with session.begin():
                requeued = 0
                # 1. processing lease 超时 + attempts 未耗尽 → pending（清锁）
                result = await session.execute(
                    text(
                        "UPDATE memory_jobs SET status='pending', locked_at=NULL, "
                        "locked_by=NULL, updated_at=:now "
                        "WHERE status='processing' AND locked_at <= :lease_cutoff "
                        "AND attempts < :max_attempts"
                    ),
                    {"now": now_naive, "lease_cutoff": lease_cutoff, "max_attempts": max_attempts},
                )
                requeued += result.rowcount or 0
                # 2. processing lease 超时 + attempts 耗尽 → dead（清锁）
                await session.execute(
                    text(
                        "UPDATE memory_jobs SET status='dead', locked_at=NULL, "
                        "locked_by=NULL, updated_at=:now "
                        "WHERE status='processing' AND locked_at <= :lease_cutoff "
                        "AND attempts >= :max_attempts"
                    ),
                    {"now": now_naive, "lease_cutoff": lease_cutoff, "max_attempts": max_attempts},
                )
                # 3. failed 到期 + attempts 未耗尽 → pending（重试）
                result = await session.execute(
                    text(
                        "UPDATE memory_jobs SET status='pending', updated_at=:now "
                        "WHERE status='failed' AND available_at <= :now_naive "
                        "AND attempts < :max_attempts"
                    ),
                    {"now": now_naive, "now_naive": now_naive, "max_attempts": max_attempts},
                )
                requeued += result.rowcount or 0
                # 4. failed 到期 + attempts 耗尽 → dead（终态，清锁）
                await session.execute(
                    text(
                        "UPDATE memory_jobs SET status='dead', locked_at=NULL, "
                        "locked_by=NULL, updated_at=:now "
                        "WHERE status='failed' AND available_at <= :now_naive "
                        "AND attempts >= :max_attempts"
                    ),
                    {"now": now_naive, "now_naive": now_naive, "max_attempts": max_attempts},
                )
        return requeued

    async def get_job_by_idempotency_key(self, key: str) -> MemoryJob | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(MemoryJobORM).where(MemoryJobORM.idempotency_key == key)
                )
            ).scalar_one_or_none()
            return _job_to_domain(row) if row is not None else None

    # ---------- 查询 ----------

    async def search_memories(
        self, query: "RecallQuery",
    ) -> list[tuple[MemoryItem, float]]:
        """两路召回（方案 18.5）：路径 A 实体精确 + 路径 B FULLTEXT ngram。

        Returns:
            [(MemoryItem, relevance)]，按 memory_id 去重，同一候选保留最高 relevance。
            路径 A 实体精确命中 relevance=1.0；路径 B 的 MATCH 分数在 Python 侧
            按本集合最大分数归一化（不同查询/语料库的原始 MATCH 分数不可横向比较）。
        """
        from agent.memory.recall import RecallQuery

        kind_values = [k.value for k in query.kinds]
        # 两路公共过滤（方案 18.5）：user/status/expires/kind
        base_conditions = [
            MemoryItemORM.user_id == query.user_id,
            MemoryItemORM.status == "active",
            MemoryItemORM.expires_at.is_(None)
            | (MemoryItemORM.expires_at > func.utc_timestamp()),
            MemoryItemORM.kind.in_(kind_values),
        ]

        scored: dict[str, tuple[MemoryItem, float]] = {}

        # 路径 A：entity_type + entity_id 精确召回（relevance = 1.0）
        if query.entity_refs:
            stmt_a = select(MemoryItemORM).where(
                *base_conditions,
                or_(
                    *[
                        and_(
                            MemoryItemORM.entity_type == ref.entity_type.value,
                            MemoryItemORM.entity_id == ref.entity_id,
                        )
                        for ref in query.entity_refs
                    ]
                ),
            )
            async with self._session_factory() as session:
                rows = (await session.execute(stmt_a)).scalars().all()
            for row in rows:
                item = _to_domain_item(row)
                scored[item.memory_id] = (item, 1.0)

        # 路径 B：FULLTEXT(content) AGAINST(query_text IN NATURAL LANGUAGE MODE)
        # （方案 18.5 固定自然语言模式；SQLAlchemy MySQL 方言的 match() 固定布尔模式，
        # 用完整 raw SQL 保证与方案 SQL 一致）
        if query.query_text:
            sql_b = """
                SELECT memory_id, user_id, kind, content, data, entity_type, entity_id,
                       source_type, source_thread_id, source_message_id, status, fingerprint,
                       created_at, updated_at, expires_at,
                       MATCH(content) AGAINST(:q IN NATURAL LANGUAGE MODE) AS ft_relevance
                FROM memory_items
                WHERE user_id = :user_id
                  AND status = 'active'
                  AND (expires_at IS NULL OR expires_at > UTC_TIMESTAMP())
                  AND kind IN :kinds
                  AND MATCH(content) AGAINST(:q IN NATURAL LANGUAGE MODE)
                ORDER BY ft_relevance DESC
                LIMIT :limit
            """
            stmt_b = text(sql_b).bindparams(
                bindparam("kinds", expanding=True),
            )
            async with self._session_factory() as session:
                result = await session.execute(
                    stmt_b,
                    {
                        "user_id": query.user_id,
                        "q": query.query_text,
                        "kinds": kind_values,
                        "limit": query.limit,
                    },
                )
                rows = result.all()
            raw = [(row, float(row[15] or 0.0)) for row in rows]
            max_ft = max((v for _, v in raw), default=0.0) or 1.0
            for row, ft_relevance in raw:
                item = _row_to_item(row)
                normalized = ft_relevance / max_ft
                prev = scored.get(item.memory_id)
                if prev is None or normalized > prev[1]:
                    scored[item.memory_id] = (item, normalized)

        return list(scored.values())

    async def list_memories(
        self, user_id: str, filters: MemoryListFilter,
        cursor: str | None, limit: int,
    ) -> MemoryPage:
        """(updated_at, memory_id) opaque cursor 分页（方案 20.2），默认只显示 active。"""
        stmt = select(MemoryItemORM).where(MemoryItemORM.user_id == user_id)
        if filters.status is None:
            stmt = stmt.where(MemoryItemORM.status == "active")
        else:
            stmt = stmt.where(MemoryItemORM.status == filters.status.value)
        if filters.kind is not None:
            stmt = stmt.where(MemoryItemORM.kind == filters.kind.value)
        if cursor is not None:
            cur_updated_at, cur_memory_id = _decode_cursor(cursor)
            # keyset 条件（review #6）：显式 or_/and_，
            # Python tuple 比较在 MySQL 方言下会丢失 memory_id tie-breaker
            stmt = stmt.where(
                or_(
                    MemoryItemORM.updated_at < cur_updated_at.replace(tzinfo=None),
                    and_(
                        MemoryItemORM.updated_at == cur_updated_at.replace(tzinfo=None),
                        MemoryItemORM.memory_id < cur_memory_id,
                    ),
                )
            )
        stmt = (
            stmt.order_by(MemoryItemORM.updated_at.desc(), MemoryItemORM.memory_id.desc())
            .limit(limit + 1)
        )

        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()

        has_more = len(rows) > limit
        page_rows = rows[:limit]
        items = [_to_domain_item(r) for r in page_rows]
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = _encode_cursor(last.updated_at, last.memory_id)
        return MemoryPage(items=items, next_cursor=next_cursor, has_more=has_more)

    async def get_memory_revision(self, user_id: str) -> int:
        async with self._session_factory() as session:
            row = await session.get(MemoryUserStateORM, user_id)
            return row.memory_revision if row is not None else 0

    async def find_memories_for_forget(
        self, user_id: str, query: str, limit: int = 21,
    ) -> list[MemoryItem]:
        """模糊 forget 专用查询（review 3.1）：只取 limit 条即可判定 0/1/>=2。

        使用 content.contains(autoescape=True) 防止用户输入的 % / _ 被解释为
        LIKE wildcard；LIMIT 21 覆盖"最多展示前 20 条 + 判定 has_more"。
        """
        if not query:
            raise InvalidMemoryData("query 不能为空")
        stmt = (
            select(MemoryItemORM)
            .where(
                MemoryItemORM.user_id == user_id,
                MemoryItemORM.status == "active",
                MemoryItemORM.content.contains(query, autoescape=True),
            )
            .limit(limit)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
        return [_to_domain_item(r) for r in rows]


# ============================================================
# Profile patch 应用与类型转换（纯函数，review A：不 mutate attached ORM row）
# ============================================================

def _profile_row_to_values(row: MemoryProfileORM) -> dict[str, Any]:
    """ORM 行 → 纯值 dict（patch 计算的输入快照）。"""
    defaults = row.procurement_defaults or {}
    return {
        "output_format": row.output_format,
        "chart_type": row.chart_type,
        "currency": row.currency,
        "language": row.language,
        "procurement_defaults": {
            "delivery_days_max": defaults.get("delivery_days_max"),
            "quality_standards": list(defaults.get("quality_standards", [])),
            "preferred_regions": list(defaults.get("preferred_regions", [])),
            "blocked_suppliers": list(defaults.get("blocked_suppliers", [])),
        },
    }


def _apply_patch_to_values(
    values: dict[str, Any], patch: ProfilePatch
) -> dict[str, Any]:
    """纯内存应用 ProfilePatch（方案 16.2 语义），返回新值 dict，不触碰 ORM。"""
    result = {
        "output_format": values["output_format"],
        "chart_type": values["chart_type"],
        "currency": values["currency"],
        "language": values["language"],
        "procurement_defaults": dict(values["procurement_defaults"]),
    }
    defaults = result["procurement_defaults"]

    for op in patch.scalar_ops:
        if op.field == "output_format":
            result["output_format"] = None if op.op == "clear" else op.value.value
        elif op.field == "chart_type":
            result["chart_type"] = None if op.op == "clear" else op.value.value
        elif op.field == "currency":
            result["currency"] = None if op.op == "clear" else op.value
        elif op.field == "language":
            result["language"] = None if op.op == "clear" else op.value
        elif op.field == "delivery_days_max":
            if op.op == "clear":
                defaults.pop("delivery_days_max", None)
            else:
                defaults["delivery_days_max"] = op.value

    for op in patch.list_ops:
        key = op.field  # quality_standards / preferred_regions / blocked_suppliers
        current = list(defaults.get(key, []))
        if op.op == "replace":
            current = list(op.values)
        elif op.op == "add":
            for v in op.values:
                if v not in current:
                    current.append(v)
        elif op.op == "remove":
            current = [v for v in current if v not in set(op.values)]
        defaults[key] = current

    return result


def _enum_value(value: Any) -> str | None:
    """输出_format/chart_type 枚举 → 数据库字符串；None 原样返回。"""
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _conflict_key(row: MemoryItemORM, field: str) -> Any:
    return (row.data or {}).get(field)


def _profile_to_domain(row: MemoryProfileORM) -> MemoryProfile:
    from agent.memory.models import ChartType, OutputFormat
    defaults = row.procurement_defaults or {}
    return MemoryProfile(
        user_id=row.user_id,
        output_format=None if row.output_format is None else OutputFormat(row.output_format),
        chart_type=None if row.chart_type is None else ChartType(row.chart_type),
        currency=row.currency,
        language=row.language,
        procurement_defaults={
            "delivery_days_max": defaults.get("delivery_days_max"),
            "quality_standards": defaults.get("quality_standards", []),
            "preferred_regions": defaults.get("preferred_regions", []),
            "blocked_suppliers": defaults.get("blocked_suppliers", []),
        },
        version=row.version,
        created_at=_utc(row.created_at),
        updated_at=_utc(row.updated_at),
    )
