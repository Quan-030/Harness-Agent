# src/agent/middlewares/memory_recall.py
"""Memory v2 模型调用期记忆召回中间件（方案 18.6）。

- abefore_agent：一次完整预取（Profile + 文本启发式触发的 Item 召回）→ invocation snapshot
- awrap_model_call：每次 model call 查 memory_revision 单行 PK，变化则重建 snapshot；
  工具循环内复用同一 HumanMessage 对应的 snapshot
- 注入只修改本次 ModelRequest：Profile 追加到 SystemMessage 末尾（保留原 prompt 的
  id/name/metadata/content blocks）；Items 包装进当前 HumanMessage 临时副本
  （记忆块在前、用户问题在后）。绝不写入 LangGraph state / checkpoint / 展示消息
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware, ModelRequest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from agent.memory.models import EntityType, MemoryKind
from agent.memory.recall import (
    CANDIDATE_LIMIT,
    EntityRef,
    MemoryInvocationSnapshot,
    QueryTextNormalizer,
    RecallIntent,
    ScoredMemory,
    apply_token_budget,
    render_profile_defaults,
    render_retrieved_items,
    rerank,
)
from agent.memory.repository import MemoryRepository

logger = logging.getLogger(__name__)

# ============================================================
# 召回触发：受控词典 + 稳定编码正则（对齐点 3：文本启发式，不依赖路由状态）
# ============================================================

_PROCUREMENT_KEYWORDS = (
    "供应商", "物料", "采购", "订单", "价格", "报价", "比价",
    "交期", "交付", "质量", "库存", "预警", "分析", "对比",
    "历史", "结果", "反馈", "询价", "下单", "采购单",
    "supplier", "part", "order", "price", "delivery",
)

_STABLE_CODE_PATTERN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{1,4}-?\d{4,}(?![A-Za-z0-9])")

# 召回范围：四类记忆（普通闲聊不触发）
_RECALL_KINDS = {
    MemoryKind.SUPPLIER_CONTEXT,
    MemoryKind.PROCUREMENT_CONSTRAINT,
    MemoryKind.TASK_OUTCOME,
    MemoryKind.USER_FEEDBACK,
}


def _ctx_get(ctx: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 与对象两种 context 形态。"""
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    return getattr(ctx, key, default)


def classify_entity_ref(code: str) -> EntityRef | None:
    """稳定编码 → 实体引用（显式前缀分类，P0 review）。

    - PO-* → order
    - MAT-*/M-* → material
    - 无法可靠分类（如 ISO9001、A1234）→ None（不伪造实体类型，
      编码仍保留在 query_text 中走全文召回路径 B）
    """
    upper = code.upper()
    if upper.startswith("PO-"):
        return EntityRef(entity_type=EntityType.ORDER, entity_id=code)
    if upper.startswith(("MAT-", "M-")):
        return EntityRef(entity_type=EntityType.MATERIAL, entity_id=code)
    return None


def _find_latest_human_message(messages: list[BaseMessage]) -> HumanMessage | None:
    """从 state 找到最新真实 HumanMessage（纯文本且拥有稳定 ID）。"""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message
    return None


def _human_message_id(human: HumanMessage) -> str:
    """稳定 ID：优先 message.id（当前用户输入已带 id），否则用内容摘要兜底。"""
    if human.id:
        return human.id
    # 无 id 时以内容哈希作为 identity（第一版；chat.py 已为输入消息分配 id）
    import hashlib
    return "h-" + hashlib.sha256(human.content.encode("utf-8")).hexdigest()[:16]


class MemoryRecallMiddleware(AgentMiddleware):
    """主 Agent 专用：Profile 每轮注入 + 采购相关 Item 按文本启发式召回注入。"""

    def __init__(
        self,
        repository: MemoryRepository,
        normalizer: QueryTextNormalizer | None = None,
    ) -> None:
        self._repository = repository
        self._normalizer = normalizer or QueryTextNormalizer()

    # ---------- 预取 ----------

    async def abefore_agent(self, state: dict[str, Any], runtime: Any) -> None:
        """一次完整预取：Profile + 启发式 Item 召回 → runtime.context.memory_snapshot。"""
        ctx = runtime.context
        user_id = _ctx_get(ctx, "user_id")
        if not user_id:
            return None

        human = _find_latest_human_message(state.get("messages") or [])
        if human is None:
            return None

        try:
            snapshot = await self.build_snapshot(
                user_id=user_id,
                thread_id=str(_ctx_get(ctx, "thread_id", "") or ""),
                human=human,
                invocation_id=str(_ctx_get(ctx, "invocation_id", "") or ""),
                resume_id=_ctx_get(ctx, "resume_id"),
                memory_revision=None,
            )
        except Exception as exc:
            # 召回失败降级：本轮无长期记忆回答（方案 5.8），记录指标，不影响主流程
            logger.warning(
                "event=memory_recall_degraded reason_code=recall_prefetch_failed "
                "outcome=degraded error_type=%s user_id=%s",
                type(exc).__name__, user_id,
            )
            return None
        ctx.memory_snapshot = snapshot
        logger.info(
            "event=memory_recall_injected outcome=injected candidate_count=%d "
            "revision=%d user_id=%s",
            len(snapshot.items), snapshot.memory_revision, user_id,
        )
        return None

    async def build_snapshot(
        self,
        *,
        user_id: str,
        thread_id: str,
        human: HumanMessage,
        invocation_id: str,
        resume_id: str | None,
        memory_revision: int | None,
    ) -> MemoryInvocationSnapshot:
        """构造 invocation snapshot。

        Profile 与 Item recall 采用不同故障边界（P0 review）：Profile 读取失败
        整体降级（abefore_agent 捕获）；Item recall 失败只降级为 profile-only
        snapshot，已成功读取的 Profile 不得被 Item 局部异常一并丢弃。
        """
        profile = await self._repository.get_or_create_profile(user_id)

        if memory_revision is None:
            memory_revision = await self._repository.get_memory_revision(user_id)

        items: list[ScoredMemory] = []
        try:
            intent = self._build_recall_intent(human.content)
            if intent is not None:
                query = RecallIntentToQuery(intent, user_id).to_query()
                candidates = await self._repository.search_memories(query)
                relevance_by_id = {item.memory_id: rel for item, rel in candidates}
                items = rerank(
                    [item for item, _ in candidates],
                    relevance_by_id=relevance_by_id,
                )
                items = apply_token_budget(items)
        except Exception as exc:
            # Item recall 局部失败 → profile-only snapshot（方案 5.8 降级语义）
            logger.error(
                "event=memory_item_recall_degraded outcome=profile_only "
                "error_type=%s user_id=%s",
                type(exc).__name__, user_id,
            )

        return MemoryInvocationSnapshot(
            user_id=user_id,
            thread_id=thread_id,
            latest_human_message_id=_human_message_id(human),
            invocation_id=invocation_id,
            resume_id=resume_id,
            memory_revision=memory_revision,
            profile=profile,
            items=items,
        )

    def _build_recall_intent(self, text: str) -> RecallIntent | None:
        """文本启发式触发（对齐点 3）：采购关键词或稳定编码命中才召回。

        稳定编码按前缀显式分类（P0 review）：只对可靠识别类型的编码生成
        EntityRef（路径 A 实体精确召回）；无法分类的编码（如 ISO9001）不伪造
        实体类型，仅保留在 query_text 中走全文召回（路径 B）。
        """
        lowered = text.lower()
        hit_keyword = any(kw in lowered for kw in _PROCUREMENT_KEYWORDS)
        codes = _STABLE_CODE_PATTERN.findall(text)

        query_text = self._normalizer.normalize(
            text, entity_terms=list(codes), task_type=None
        )
        if not hit_keyword and not codes:
            return None
        if not query_text and not codes:
            return None

        entity_refs: list[EntityRef] = []
        for code in codes:
            ref = classify_entity_ref(code)
            if ref is not None:
                entity_refs.append(ref)
        return RecallIntent(
            kinds=set(_RECALL_KINDS),
            entity_refs=entity_refs[:20],
            query_text=query_text,
            limit=CANDIDATE_LIMIT,
        )

    # ---------- model call 包装 ----------

    async def awrap_model_call(self, request: ModelRequest, handler):
        ctx = request.runtime.context
        user_id = _ctx_get(ctx, "user_id")
        if not user_id:
            return await handler(request)

        messages = list(request.messages)
        current_human = _find_latest_human_message(messages)
        snapshot = _ctx_get(ctx, "memory_snapshot")

        try:
            current_revision = await self._repository.get_memory_revision(user_id)
        except Exception as exc:
            # review #6：revision 无法确认 → 不使用任何 Memory（清掉旧 snapshot），
            # 否则 forget/delete-all 提交后因临时故障复用旧 snapshot 会重新注入已删除记忆
            # 只记录安全摘要（方案 20.3：不输出可能含 SQL 参数的 raw exception）
            logger.error(
                "event=memory_recall_degraded reason_code=revision_check_failed "
                "outcome=degraded error_type=%s user_id=%s",
                type(exc).__name__, user_id,
            )
            ctx.memory_snapshot = None
            return await handler(request)

        if not self._snapshot_matches(snapshot, request, current_human, current_revision):
            if current_human is not None:
                try:
                    snapshot = await self.build_snapshot(
                        user_id=user_id,
                        thread_id=str(_ctx_get(ctx, "thread_id", "") or ""),
                        human=current_human,
                        invocation_id=str(_ctx_get(ctx, "invocation_id", "") or ""),
                        resume_id=_ctx_get(ctx, "resume_id"),
                        memory_revision=current_revision,
                    )
                except Exception as exc:
                    # review #6：rebuild 失败 → 清 snapshot，无记忆继续（fail-open 闭合）
                    logger.error(
                        "event=memory_recall_degraded reason_code=snapshot_rebuild_failed "
                        "outcome=degraded error_type=%s user_id=%s",
                        type(exc).__name__, user_id,
                    )
                    snapshot = None
                ctx.memory_snapshot = snapshot

        if snapshot is None:
            return await handler(request)

        # Profile → SystemMessage 末尾（保留原 prompt 的 id/name/metadata/content blocks）
        dynamic_system = _append_profile_preserving_message(
            request.system_message, render_profile_defaults(snapshot.profile)
        )

        # Items → 当前 HumanMessage 临时副本（记忆块在前，用户问题在后）
        if current_human is not None and snapshot.items:
            augmented = render_retrieved_items(snapshot.items, current_human.content)
            messages[messages.index(current_human)] = current_human.model_copy(
                update={"content": augmented}
            )

        return await handler(
            request.override(system_message=dynamic_system, messages=messages)
        )

    def _snapshot_matches(
        self,
        snapshot: Any,
        request: ModelRequest,
        current_human: HumanMessage | None,
        current_revision: int | None,
    ) -> bool:
        """snapshot 身份键：user + thread + human_id + invocation + resume + revision。"""
        if snapshot is None or current_human is None:
            return False
        if current_revision is not None and snapshot.memory_revision != current_revision:
            return False
        if snapshot.latest_human_message_id != _human_message_id(current_human):
            return False
        ctx = request.runtime.context
        if snapshot.invocation_id != str(_ctx_get(ctx, "invocation_id", "") or ""):
            return False
        if snapshot.resume_id != _ctx_get(ctx, "resume_id"):
            return False
        return True


class RecallIntentToQuery:
    """调用层 Intent → 带 scope 的 RecallQuery（方案 18.5：Service 补充 user_id）。"""

    def __init__(self, intent: RecallIntent, user_id: str) -> None:
        self._intent = intent
        self._user_id = user_id

    def to_query(self):
        from agent.memory.recall import RecallQuery
        return RecallQuery(
            user_id=self._user_id,
            kinds=self._intent.kinds,
            entity_refs=self._intent.entity_refs,
            query_text=self._intent.query_text,
            limit=self._intent.limit,
        )


def _append_profile_preserving_message(
    system_message: SystemMessage | None,
    profile_block: str | None,
) -> SystemMessage | None:
    """Profile 区块追加到 SystemMessage 末尾（方案 18.6）。

    保留原 SystemMessage 的 id/name/additional_kwargs/response_metadata 和
    content blocks，不得粗暴重建而丢失 provider metadata。
    """
    if profile_block is None:
        return system_message
    if system_message is None:
        return SystemMessage(content=profile_block)
    if isinstance(system_message.content, str):
        new_content = system_message.content + "\n\n" + profile_block
    else:
        # content blocks 形态：追加一个 text block
        blocks = list(system_message.content)
        new_content = blocks + [{"type": "text", "text": profile_block}]
    return SystemMessage(
        content=new_content,
        id=system_message.id,
        name=system_message.name,
        additional_kwargs=system_message.additional_kwargs,
        response_metadata=system_message.response_metadata,
    )
