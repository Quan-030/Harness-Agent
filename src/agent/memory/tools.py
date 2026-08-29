# src/agent/memory/tools.py
"""Agent 受控记忆工具（方案 20.1）。

- remember / update_preference / forget_memory：同步执行，提交成功后才返回确认
- 工具 schema 不暴露 actor/source/status/时间/目标用户参数；
  user_id 由工厂闭包从 create_main_agent 的 config 注入（与 assign_skill 同模式）
- 所有回执使用实际事务结果，禁止模型在工具成功前声称完成
"""
from __future__ import annotations

import logging

from langchain_core.tools import tool

from agent.memory.models import (
    EntityType,
    MemoryKind,
    ProfilePatch,
)
from agent.memory.policies import SensitiveContentError
from agent.memory.repository import (
    ActorContext,
    InvalidMemoryData,
    InvalidMemoryState,
    InvalidProfilePatch,
    ProfileVersionConflict,
)
from agent.memory.service import MemoryService

logger = logging.getLogger(__name__)

# review 3.3：预期业务异常 → 明确用户可见文案
_EXPECTED_ERRORS = (
    SensitiveContentError,
    ProfileVersionConflict,
    InvalidMemoryData,
    InvalidMemoryState,
    InvalidProfilePatch,
)


def _safe_failure(operation: str, exc: Exception) -> str:
    """未知异常：只记录异常类型（不打印 str(exc)/traceback，避免 SQL/DSN/正文泄露），
    返回通用文案（review 3.3）。"""
    logger.error("event=memory_write_failed outcome=failed tool=%s error_type=%s", operation, type(exc).__name__)
    return "记忆服务暂不可用，请稍后重试。"


def _infer_kind(entity_type: EntityType | None) -> MemoryKind:
    """remember 未指定 kind 时的推断（review #4 收紧）：

    - supplier → supplier_context（data 字段全可选，缺省 data={} 合法）
    - material/order → procurement_constraint（需要 data.constraint_name+value）
    - 无实体 → 不推断（user_feedback/task_outcome 的 data 有必填字段，
      要求调用方显式提供 kind + data，避免必然的 ValidationError）
    """
    if entity_type == EntityType.SUPPLIER:
        return MemoryKind.SUPPLIER_CONTEXT
    if entity_type in (EntityType.MATERIAL, EntityType.ORDER):
        return MemoryKind.PROCUREMENT_CONSTRAINT
    raise ValueError(
        "未指定 kind 且无 supplier 实体时无法推断记忆类型；"
        "请显式提供 kind（supplier_context / procurement_constraint / "
        "task_outcome / user_feedback）并按要求提供 data 必填字段"
    )


def create_memory_tools(service: MemoryService, user_id: str, thread_id: str = "") -> list:
    """创建受控记忆工具（按用户注入 scope，与 create_assign_skill_tool 同模式）。

    user_id/thread_id 来自请求 config（方案 8.1：工具 schema 不暴露目标用户参数）。
    """

    @tool
    async def remember(
        content: str,
        kind: str | None = None,
        data: dict | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        long_term: bool = False,
    ) -> str:
        """
        记住一条用户明确要求的信息（同步写入，成功后才返回确认）。

        Args:
            content: 要记住的事实内容（1~2000 字符）
            kind: 记忆类型：supplier_context / procurement_constraint /
                  task_outcome / user_feedback。缺省仅按 supplier 实体推断；
                  无实体时必须显式提供 kind
            data: kind 对应的结构化数据（按白名单校验，不允许任意键）：
                - supplier_context: {relationship?, region?, note_type?}
                  （capability/risk/relationship/other，均可选）
                - procurement_constraint: {constraint_name: 必填, value: 必填,
                  unit?}（constraint_name: delivery_days_max/amount_limit/currency/
                  quality_standard/region/supplier_restriction/other）
                - task_outcome: {task_type: 必填, result_status: 必填
                  （succeeded/failed/cancelled/partial）, selected_entity_id?}
                - user_feedback: {target_type: 必填（response/supplier/material/
                  order/workflow）, feedback_type: 必填（positive/negative/
                  correction/preference）, target_id?}
            entity_type: 关联实体类型：supplier / material / order（可选）
            entity_id: 关联实体 ID（提供 entity_type 时必须提供）
            long_term: True 表示用户明确要求长期记住（不自动过期）

        Returns:
            写入确认或错误说明（基于实际事务结果）。
        """
        try:
            kind_enum = MemoryKind(kind) if kind else _infer_kind(
                EntityType(entity_type) if entity_type else None
            )
            entity_enum = EntityType(entity_type) if entity_type else None
            if (entity_enum is None) != (entity_id is None):
                return "错误：entity_type 与 entity_id 必须同时提供或同时省略"
            result = await service.remember(
                user_id,
                ActorContext(actor_type="user", source_thread_id=thread_id),
                kind=kind_enum,
                content=content,
                data=data,
                entity_type=entity_enum,
                entity_id=entity_id,
                long_term=long_term,
            )
        except SensitiveContentError as exc:
            return f"拒绝保存：内容命中敏感信息禁存规则（{exc.reason_code}）"
        except InvalidMemoryData:
            return "拒绝保存：data 不符合该记忆类型的契约，请检查必填字段。"
        except _EXPECTED_ERRORS:
            return "保存失败：请稍后重试或检查输入。"
        except Exception as exc:
            return _safe_failure("remember", exc)

        if result.outcome == "duplicate":
            return "该信息已存在，已刷新时间戳。"
        if result.outcome == "conflict":
            return "拒绝保存：已有更高优先级的矛盾信息。"
        suffix = "（已替代旧信息）" if result.outcome == "superseded" else ""
        return f"已记住{suffix}。"

    @tool
    async def update_preference(
        patch: dict,
        expected_version: int | None = None,
    ) -> str:
        """
        更新用户默认偏好（同步写入，只允许白名单字段）。

        Args:
            patch: 偏好变更，形如
                {"scalar_ops": [{"field": "currency", "op": "set", "value": "CNY"}],
                 "list_ops": [{"field": "blocked_suppliers", "op": "add",
                               "values": ["博世"]}]}
                标量字段：output_format / chart_type / currency / language /
                delivery_days_max（op: set/clear）
                列表字段：quality_standards / preferred_regions / blocked_suppliers
                （op: replace/add/remove）
            expected_version: 当前 Profile 版本（可选；缺失时自动读取）

        Returns:
            更新确认或错误说明。
        """
        try:
            profile_patch = ProfilePatch.model_validate(patch)
            profile = await service.update_preference(
                user_id, profile_patch,
                ActorContext(actor_type="user", source_thread_id=thread_id),
                expected_version=expected_version,
            )
        except ProfileVersionConflict:
            return "保存失败：偏好已被其他会话修改，请重试。"
        except InvalidProfilePatch:
            return "保存失败：偏好变更后状态不合法（如列表超出上限）。"
        except _EXPECTED_ERRORS:
            return "保存失败：请稍后重试或检查输入。"
        except Exception as exc:
            return _safe_failure("update_preference", exc)
        return f"偏好已更新（版本 {profile.version}）。"

    @tool
    async def forget_memory(
        memory_id: str | None = None,
        query: str | None = None,
        reason: str | None = None,
    ) -> str:
        """
        忘记一条记忆（同步执行，提交成功后才返回确认）。

        Args:
            memory_id: 精确记忆 ID（可选）
            query: 模糊搜索词（可选）；命中多条时不执行，需用户明确选择
            reason: 忘记原因（可选，写入审计）

        Returns:
            操作结果说明。
        """
        try:
            result = await service.forget_memory(
                user_id, ActorContext(actor_type="user", source_thread_id=thread_id),
                memory_id=memory_id, query=query, reason=reason,
            )
        except InvalidMemoryData:
            return "操作失败：忘记条件无效（query 不能为空）。"
        except _EXPECTED_ERRORS:
            return "操作失败：请稍后重试或检查输入。"
        except Exception as exc:
            return _safe_failure("forget_memory", exc)

        if result.outcome == "forgotten":
            return "已忘记。"
        if result.outcome == "not_found":
            return "未找到匹配的记忆。"
        return (
            "找到多条匹配的记忆，请明确要忘记哪一条：\n"
            + "\n".join(f"- {item.content[:100]}" for item in result.candidates)
        )

    return [remember, update_preference, forget_memory]
