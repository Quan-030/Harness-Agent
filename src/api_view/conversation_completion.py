# src/api_view/conversation_completion.py
"""ConversationCompletionService（方案 6.1：HTTP/SSE 共用的完成持久化与唯一入队边界）。

- 唯一的自动入队所有者：由 chat.py 的 HTTP/SSE 公共完成路径调用，
  不能由 Agent 或任意 LangGraph middleware 入队
- 先确认本轮 user_message_id / assistant_message_id / checkpoint_id，
  再调用 MemoryService.enqueue_extract_job（幂等，19.2）
- 三者任一未能确认时不得创建不完整 Job（记录错误指标，留给 reconciliation 补建）
"""
from __future__ import annotations

import logging
from typing import Any

from agent.memory.repository import EnqueueJobCommand, MemoryRepository

logger = logging.getLogger(__name__)

EXTRACTOR_VERSION = "memory-v2.1"


def extract_ids_from_display_messages(
    display_messages: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    """从展示消息中提取本轮 user_message_id / assistant_message_id。

    display_messages 的 id 由 chat.py 写入（user-{uuid} / assistant-{uuid}，方案 6.1）。
    取当前轮最新一条 user 消息与最新一条 assistant 消息的 id。
    """
    user_message_id: str | None = None
    assistant_message_id: str | None = None
    for dm in display_messages:
        if dm.get("role") == "user" and dm.get("id"):
            user_message_id = dm["id"]
        elif dm.get("role") == "assistant" and dm.get("id") and dm.get("content"):
            assistant_message_id = dm["id"]
    return user_message_id, assistant_message_id


class ConversationCompletionService:
    """对话完成后的记忆抽取入队服务（HTTP/SSE 唯一入口）。"""

    def __init__(self, repository: MemoryRepository) -> None:
        self._repository = repository

    async def enqueue_extract(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_message_id: str,
        assistant_message_id: str,
        checkpoint_id: str,
    ) -> str | None:
        """确认三类 ID 后入队 extract_memory（幂等：同 key 重复入队返回既有 job）。

        Returns:
            job_id；ID 缺失时不创建不完整 Job，返回 None（留 reconciliation 补建）。
        """
        if not (user_message_id and assistant_message_id and checkpoint_id):
            logger.error(
                "event=memory_enqueue_failed reason_code=incomplete_ids "
                "outcome=skipped user=%s assistant=%s checkpoint=%s",
                bool(user_message_id), bool(assistant_message_id), bool(checkpoint_id),
            )
            return None

        payload = {
            "checkpoint_id": checkpoint_id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "extractor_version": EXTRACTOR_VERSION,
            "memory_generation": 0,  # 入队事务内由 Repository 覆盖为当前 generation
            "replay_generation": 0,
        }
        job_id = await self._repository.enqueue_job(
            EnqueueJobCommand(
                user_id=user_id,
                thread_id=thread_id,
                job_type="extract_memory",
                payload=payload,
            )
        )
        logger.info(
            "event=memory_enqueued outcome=succeeded job_id=%s thread_id=%s",
            job_id, thread_id,
        )
        return job_id
