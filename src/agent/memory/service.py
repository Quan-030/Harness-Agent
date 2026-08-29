# src/agent/memory/service.py
"""Memory v2 用例层（方案 18.1 / 20.1 / 20.3）。

MemoryService 职责：请求 scope 传递、业务规则（敏感/TTL/冲突）、
事务编排和错误映射。MiddleWare/API/tools/worker 只调用 Service，
不直接使用 ORM Session（方案 18.1 分层职责）。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.memory.models import (
    EntityType,
    MemoryItem,
    MemoryKind,
    MemoryProfile,
    MemoryStatus,
    ProfilePatch,
    SourceType,
    STRICT_MODEL_CONFIG,
    compute_fingerprint,
    new_uuid7,
)
from agent.memory.policies import MemoryPolicy, SensitiveContentError
from agent.memory.repository import (
    ActorContext,
    CreateMemoryCommand,
    InvalidMemoryData,
    MemoryListFilter,
    MemoryNotFound,
    MemoryPage,
    MemoryRepository,
    MemoryWriteResult,
    ProfileNotFound,
    ProfileVersionConflict,
)


class ForgetResult(BaseModel):
    """forget_memory 结果（方案 20.1：精确 ID / 模糊 query 0-1-多语义）。"""

    model_config = STRICT_MODEL_CONFIG
    outcome: Literal["forgotten", "not_found", "ambiguous"]
    memory_ids: list[str] = Field(default_factory=list)
    candidates: list[MemoryItem] = Field(default_factory=list, max_length=20)
    match_count: int = 0
    has_more: bool = False


def _patch_fields_changed(
    base: MemoryProfile, latest: MemoryProfile, patch: ProfilePatch,
) -> bool:
    """patch 涉及的字段在 base/latest 之间是否发生变化（review B）。

    列表字段按整个 field 判断，不尝试合并 add/remove。
    """
    scalar_getters = {
        "output_format": lambda p: p.output_format,
        "chart_type": lambda p: p.chart_type,
        "currency": lambda p: p.currency,
        "language": lambda p: p.language,
        "delivery_days_max": lambda p: p.procurement_defaults.delivery_days_max,
    }
    list_getters = {
        "quality_standards": lambda p: p.procurement_defaults.quality_standards,
        "preferred_regions": lambda p: p.procurement_defaults.preferred_regions,
        "blocked_suppliers": lambda p: p.procurement_defaults.blocked_suppliers,
    }
    for op in patch.scalar_ops:
        if scalar_getters[op.field](base) != scalar_getters[op.field](latest):
            return True
    for op in patch.list_ops:
        if list_getters[op.field](base) != list_getters[op.field](latest):
            return True
    return False


class MemoryService:
    """记忆业务用例入口。所有方法以 user_id 为 scope，调用方从 runtime 传递。"""

    def __init__(self, repository: MemoryRepository, policy: MemoryPolicy):
        self._repository = repository
        self._policy = policy

    # ---------- Profile ----------

    async def get_profile(self, user_id: str) -> MemoryProfile:
        return await self._repository.get_or_create_profile(user_id)

    async def update_preference(
        self,
        user_id: str,
        patch: ProfilePatch,
        actor: ActorContext,
        expected_version: int | None = None,
    ) -> MemoryProfile:
        """更新 Profile（乐观锁）。

        - expected_version 缺失（工具/内部路径）：先 get_or_create Profile 并读版本，
          冲突后重读重试一次（方案 20.1：不允许无限重试）
        - expected_version 提供（API 路径）：直接乐观锁更新，冲突向上传播（API 返回 409）
        """
        if expected_version is None:
            base = await self._repository.get_or_create_profile(user_id)
            try:
                return await self._repository.patch_profile(
                    user_id, patch, expected_version=base.version, actor=actor
                )
            except ProfileVersionConflict:
                # 区分同字段/不同字段冲突（review B）：
                # patch 涉及字段在 base 与 latest 之间未变化 → 竞争事务改的是其他字段
                #   → 允许基于 latest 重试一次（不同字段并发最终都保留）
                # 已变化 → 同字段并发冲突 → 直接抛（禁止 silent last-write-wins）
                latest = await self._repository.get_or_create_profile(user_id)
                if _patch_fields_changed(base, latest, patch):
                    raise
                return await self._repository.patch_profile(
                    user_id, patch, expected_version=latest.version, actor=actor
                )
        return await self._repository.patch_profile(
            user_id, patch, expected_version=expected_version, actor=actor
        )

    # ---------- Memory Item ----------

    async def remember(
        self,
        user_id: str,
        actor: ActorContext,
        *,
        kind: MemoryKind,
        content: str,
        data: dict[str, Any] | None = None,
        entity_type: EntityType | None = None,
        entity_id: str | None = None,
        source_type: SourceType = SourceType.USER_EXPLICIT,
        source_message_id: str | None = None,
        long_term: bool = False,
    ) -> MemoryWriteResult:
        """同步写入记忆（用户显式 remember / 工具验证事件共用）。

        敏感信息命中时整条拒绝（SensitiveContentError，方案 20.3）。
        TTL 默认按 kind（方案 18.4）；long_term=True 表示用户明确要求长期记住
        （expires_at=None），组织合规上限不由 LLM 自行延长。
        """
        # v1 安全边界（review 评论二）：model_inferred 禁止自动入库，
        # 仅 user_explicit（工具路径）和 tool_verified（未来 worker 确定性事件）可持久化
        if source_type == SourceType.MODEL_INFERRED:
            raise ValueError(
                "v1 不允许 model_inferred 自动入库；仅 user_explicit 和 tool_verified 可持久化"
            )

        data = {} if data is None else data
        command = self.prepare_memory_command(
            user_id=user_id,
            actor=actor,
            kind=kind,
            content=content,
            data=data,
            entity_type=entity_type,
            entity_id=entity_id,
            source_type=source_type,
            source_message_id=source_message_id,
            long_term=long_term,
        )
        return await self._repository.create_or_resolve_memory(command, actor)

    def prepare_memory_command(
        self,
        *,
        user_id: str,
        actor: ActorContext,
        kind: MemoryKind,
        content: str,
        data: dict[str, Any] | None = None,
        entity_type: EntityType | None = None,
        entity_id: str | None = None,
        source_type: SourceType = SourceType.USER_EXPLICIT,
        source_message_id: str | None = None,
        long_term: bool = False,
    ) -> CreateMemoryCommand:
        """纯计算：Policy 校验（敏感/TTL）+ fingerprint + 命令构造，**不访问数据库**。

        review #1.3：worker 自动路径先 prepare 全部命令，再在同一 MySQL 事务内
        批量提交（apply_extract_job），确保 Policy/领域规则仍在 Service 层生效。
        """
        # v1 安全边界（review 评论二）：model_inferred 禁止自动入库
        if source_type == SourceType.MODEL_INFERRED:
            raise ValueError(
                "v1 不允许 model_inferred 自动入库；仅 user_explicit 和 tool_verified 可持久化"
            )

        data = {} if data is None else data
        ttl = self._policy.validate_for_storage(content, data, kind)

        expires_at = None
        if not long_term and ttl is not None:
            expires_at = datetime.now(timezone.utc) + ttl

        try:
            return CreateMemoryCommand(
                user_id=user_id,
                memory_id=new_uuid7(),
                kind=kind,
                content=content,
                data=data,
                entity_type=entity_type,
                entity_id=entity_id,
                source_type=source_type,
                source_thread_id=actor.source_thread_id or "",
                source_message_id=source_message_id,
                fingerprint=compute_fingerprint(kind, entity_type, entity_id, content, data),
                expires_at=expires_at,
            )
        except Exception as exc:
            # 客户端 data 错误（review 3.2）→ InvalidMemoryData（API 400），
            # 不把底层 ValidationError 泄漏到 API/工具
            raise InvalidMemoryData(f"data 不符合 {kind.value} 契约: {exc}") from exc

    async def forget_memory(
        self,
        user_id: str,
        actor: ActorContext,
        *,
        memory_id: str | None = None,
        query: str | None = None,
        reason: str | None = None,
    ) -> ForgetResult:
        """忘记记忆（方案 20.1）：

        - 精确 memory_id → 直接 forget
        - 模糊 query → 命中 0 条报告 not_found；1 条执行；多条返回 ambiguous 候选
        """
        if memory_id is not None:
            try:
                await self._repository.forget_memory(user_id, memory_id, reason, actor)
            except MemoryNotFound:
                return ForgetResult(outcome="not_found")
            return ForgetResult(outcome="forgotten", memory_ids=[memory_id])

        if query is None or not query.strip():
            # review 3.1：空 query 会命中所有记忆（"" in content 恒真），必须拒绝
            raise InvalidMemoryData("query 不能为空")

        # 模糊匹配（review 3.1）：专用查询只取 limit 条即可判定 0/1/>=2，
        # 不做全表扫描；绝不能"只看第一页后猜唯一"再执行破坏性删除
        matches = await self._repository.find_memories_for_forget(
            user_id, query.strip(), limit=21
        )
        if not matches:
            return ForgetResult(outcome="not_found")
        if len(matches) > 1:
            return ForgetResult(
                outcome="ambiguous",
                candidates=matches[:20],  # ForgetResult.candidates max_length=20
                match_count=len(matches),
                has_more=len(matches) > 20,  # 21 条 → 还有更多
            )
        target = matches[0]
        await self._repository.forget_memory(user_id, target.memory_id, reason, actor)
        return ForgetResult(outcome="forgotten", memory_ids=[target.memory_id])

    async def correct_memory(
        self,
        user_id: str,
        memory_id: str,
        actor: ActorContext,
        *,
        content: str,
        data: dict[str, Any] | None = None,
    ) -> MemoryWriteResult:
        """纠正记忆（方案 20.2 PATCH 语义）：创建新 Item 并显式 supersede 旧 Item。

        新 Item 继承旧 Item 的 kind/entity，来源为 user_explicit。
        """
        old = await self._repository.get_memory(user_id, memory_id)
        # review O：显式 {} 视为"清空 data 的纠正"，只有 None 表示"未提供"
        if data is None:
            data = old.data

        ttl = self._policy.validate_for_storage(content, data, old.kind)
        expires_at = None
        if ttl is not None:
            expires_at = datetime.now(timezone.utc) + ttl

        try:
            command = CreateMemoryCommand(
                user_id=user_id,
                memory_id=new_uuid7(),
                kind=old.kind,
                content=content,
                data=data,
                entity_type=old.entity_type,
                entity_id=old.entity_id,
                source_type=SourceType.USER_EXPLICIT,
                source_thread_id=actor.source_thread_id or old.source_thread_id,
                fingerprint=compute_fingerprint(old.kind, old.entity_type, old.entity_id, content, data),
                expires_at=expires_at,
                supersede_target_id=old.memory_id,
            )
        except Exception as exc:
            # 客户端 data 错误（review 3.2）→ InvalidMemoryData（API 400）
            raise InvalidMemoryData(
                f"data 不符合 {old.kind.value} 契约: {exc}"
            ) from exc
        return await self._repository.create_or_resolve_memory(command, actor)

    async def delete_all_memories(self, user_id: str, actor: ActorContext) -> None:
        """删除全部记忆（方案 20.4）：generation/revision 递增、Profile 删除、
        Items 停止召回、pending/failed job 取消。调用方须先完成二次确认。"""
        await self._repository.delete_all_memories(user_id, actor)

    # ---------- 查询 ----------

    async def list_memories(
        self,
        user_id: str,
        filters: MemoryListFilter | None = None,
        cursor: str | None = None,
        limit: int = 50,
    ) -> MemoryPage:
        return await self._repository.list_memories(
            user_id, filters or MemoryListFilter(), cursor, limit
        )
