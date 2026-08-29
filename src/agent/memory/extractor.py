# src/agent/memory/extractor.py
"""Memory v2 structured extractor（方案 4.3 / 6.3 / 18.3）。

- 只给抽取器本轮必要信息：最后一条用户消息、最终 Agent 回复、工具结果摘要、
  当前 Profile 和可能冲突的少量已有记忆（6.2，不送整个累计 thread）
- 模型使用 structured output 返回 MemoryExtractionResult（4.3）
- 抽取阈值（18.3）：>= 0.85 进入规则校验；[0.65, 0.85) 与 < 0.65 一律丢弃
- 第一版禁止 model_inferred 自动入库（6.4）；用户显式 remember 走同步路径
- 确定性工具事件（6.3 第 1 条）：成功工具事件 → tool_verified 候选（置信度 1.0）
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from agent.memory.models import (
    MemoryCandidate,
    MemoryExtractionResult,
    MemoryKind,
)
from agent.memory.policies import EXTRACTION_CONFIDENCE_MIN, EXTRACTION_CONFIDENCE_PERSIST

logger = logging.getLogger(__name__)


# ============================================================
# 抽取阈值过滤（方案 18.3）
# ============================================================

def apply_extraction_threshold(
    result: MemoryExtractionResult,
) -> MemoryExtractionResult:
    """按置信度过滤候选（18.3）：

    >= 0.85 保留（进入规则校验）；[0.65, 0.85) 与 < 0.65 丢弃。
    丢弃时不保存候选正文，只记录聚合指标（本函数返回过滤后结果）。
    """
    kept = [
        c for c in result.memory_candidates
        if c.extraction_confidence >= EXTRACTION_CONFIDENCE_PERSIST
    ]
    dropped = len(result.memory_candidates) - len(kept)
    if dropped:
        logger.info("event=memory_candidate_dropped reason_code=below_confidence_threshold outcome=dropped count=%s", dropped)
    return MemoryExtractionResult(
        profile_patches=result.profile_patches,
        memory_candidates=kept,
    )


# ============================================================
# 确定性工具事件候选（方案 6.3 第 1 条：工具成功事件直接形成候选记忆）
# ============================================================

# 工具成功事件 → (kind, data) 规则（第一版：订单/询价类成功事件 → task_outcome）
_TOOL_EVENT_RULES: dict[str, dict[str, Any]] = {
    "order_create": {
        "kind": MemoryKind.TASK_OUTCOME,
        "task_type": "订单创建",
        "result_status": "succeeded",
    },
    "order_update": {
        "kind": MemoryKind.TASK_OUTCOME,
        "task_type": "订单修改",
        "result_status": "succeeded",
    },
}


def tool_event_candidates(
    tool_results: Sequence[dict[str, Any]],
) -> list[MemoryCandidate]:
    """从工具结果摘要确定性生成 tool_verified 候选（置信度 1.0）。

    tool_results 每项形如 {"tool_name": "order_create", "summary": "...", "succeeded": True}。

    review（PR #15 第二轮）：**succeeded 必须显式为 True 才生成候选**——
    来自工具执行层的结构化状态（ToolMessage.status / envelope），
    失败或无法确认状态一律不生成（防止业务失败被记成最高可信度的"成功历史事实"）。
    禁止从 summary 文本猜测成功/失败。
    """
    candidates: list[MemoryCandidate] = []
    for result in tool_results:
        if result.get("succeeded") is not True:
            # 失败 / 无法确认状态 → 不生成 deterministic candidate
            continue
        tool_name = result.get("tool_name", "")
        rule = _TOOL_EVENT_RULES.get(tool_name)
        if rule is None:
            continue
        summary = str(result.get("summary", "")).strip()
        content = f"{rule['task_type']}成功：{summary}" if summary else f"{rule['task_type']}成功"
        candidates.append(
            MemoryCandidate(
                kind=rule["kind"],
                content=content[:2000],
                data={
                    "task_type": rule["task_type"],
                    "result_status": rule["result_status"],
                },
                extraction_confidence=1.0,  # 确定性事件
            )
        )
    return candidates


# ============================================================
# 模型抽取（方案 6.3 第 2-4 条）
# ============================================================

class StructuredExtractor:
    """结构化抽取器：模型 structured output → MemoryExtractionResult。"""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    async def extract(
        self,
        *,
        user_message: str,
        assistant_message: str,
        tool_results_summary: str = "",
        profile_context: str = "",
    ) -> MemoryExtractionResult:
        """抽取本轮记忆候选（输入只含本轮必要信息，6.2）。

        Raises:
            ValidationError: 模型返回结构无法通过 MemoryExtractionResult 校验
                （方案 6.3 第 4 条：Pydantic 校验失败整批拒绝，由调用方隔离）。
        """
        prompt = self._build_prompt(
            user_message=user_message,
            assistant_message=assistant_message,
            tool_results_summary=tool_results_summary,
            profile_context=profile_context,
        )
        try:
            structured = self._model.with_structured_output(MemoryExtractionResult)
            raw = await structured.ainvoke(prompt)
        except ValidationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"模型抽取调用失败：{type(exc).__name__}") from exc

        if not isinstance(raw, MemoryExtractionResult):
            raw = MemoryExtractionResult.model_validate(raw)
        return raw

    @staticmethod
    def _build_prompt(
        *,
        user_message: str,
        assistant_message: str,
        tool_results_summary: str,
        profile_context: str,
    ) -> str:
        parts = [
            "从以下对话中提取对未来有价值、可复用的用户记忆。",
            "规则：",
            "- 只提取用户明确陈述的偏好、约束、供应商事实、任务结果和反馈；",
            "  一次性数字和普通查询条件默认不记。",
            "- 抽取置信度 extraction_confidence 表示'本次抽取是否准确'（0~1）；",
            f"  低于 {EXTRACTION_CONFIDENCE_PERSIST} 的候选会被丢弃。",
            "- 同一事实只能进入 profile_patches 或 memory_candidates 之一，禁止重复。",
            "- 不要生成 user_id、memory_id、时间、状态或来源字段。",
            f"当前 Profile 摘要：{profile_context or '（无）'}",
            f"用户消息：{user_message}",
            f"最终回复：{assistant_message}",
            f"工具结果摘要：{tool_results_summary or '（无）'}",
        ]
        return "\n".join(parts)
