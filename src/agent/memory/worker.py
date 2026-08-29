# src/agent/memory/worker.py
"""Memory v2 durable outbox worker（方案 19.3 / 19.4）。

- 每次领取 20 条（SKIP LOCKED），处理前按 19.3 事务协议比较 generation
- 默认最大尝试 5 次；退避 min(300, 2**attempts) + random(0,1) 秒
- processing lease 默认 5 分钟；reaper 恢复超时任务
- MEMORY_WORKER_MODE=disabled|embedded|standalone（19.3）
- 生成检查不匹配 → cancelled（不写 Profile/Item/Event，不允许 replay）
"""
from __future__ import annotations

import asyncio
import inspect
import logging
import os
import random
import uuid
from datetime import datetime, timedelta, timezone

from agent.memory.extractor import tool_event_candidates
from agent.memory.models import SourceType
from agent.memory.repository import ActorContext, MemoryJob, MemoryRepository

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_LEASE_SECONDS = 300
DEFAULT_POLL_INTERVAL = 1.0
MAX_POLL_INTERVAL = 5.0
DEFAULT_CLAIM_LIMIT = 20
SHUTDOWN_GRACE_SECONDS = 30


def _backoff_seconds(attempts: int) -> float:
    """方案 19.3：min(300, 2 ** attempts) + random(0, 1) 秒。"""
    return min(300, 2 ** attempts) + random.random()


class JobGenerationMismatch(Exception):
    """job payload 的 generation 与当前 memory_user_state 不一致（delete-all 后）。"""


class ReferencedMessageUnavailable(RuntimeError):
    """本轮引用的 user/assistant 消息缺失或顺序异常（方案 19.4：重试 → 超窗 dead）。"""


class JobLeaseLost(RuntimeError):
    """job 的 lease 已被 reaper 回收、由其他 worker 接手（stale worker 不得写入）。"""


def safe_error_summary(exc: Exception) -> str:
    """异常安全摘要（方案 20.3：job error 不得包含敏感正文）。

    不保存 str(exc)——SQLAlchemy 异常可能含 SQL 与 bound parameters（记忆正文/
    data/query_text），Pydantic ValidationError 含 input_value。
    只保留受控 reason code + 异常类型名。
    """
    from pydantic import ValidationError
    from sqlalchemy.exc import OperationalError

    if isinstance(exc, ReferencedMessageUnavailable):
        reason_code = "referenced_message_unavailable"
    elif isinstance(exc, JobGenerationMismatch):
        reason_code = "generation_mismatch"
    elif isinstance(exc, ValidationError):
        reason_code = "validation_failed"
    elif isinstance(exc, OperationalError):
        reason_code = "db_unavailable"
    else:
        reason_code = "unexpected_error"
    return f"reason_code={reason_code} error_type={type(exc).__name__}"


class MemoryWorker:
    """消费 extract_memory / expire_memory job 的后台 worker。"""

    def __init__(
        self,
        repository: MemoryRepository,
        *,
        worker_id: str | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        claim_limit: int = DEFAULT_CLAIM_LIMIT,
        memory_service: Any | None = None,
        display_messages_loader: Any | None = None,
        extractor: Any | None = None,
    ) -> None:
        self._repository = repository
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._max_attempts = max_attempts
        self._lease_seconds = lease_seconds
        self._poll_interval = poll_interval
        self._claim_limit = claim_limit
        # 第 10 步接入：memory_service 写记忆、display_messages_loader 读消息、
        # extractor 模型抽取（第一版自动路径禁用模型候选，6.4）
        self._memory_service = memory_service
        self._display_messages_loader = display_messages_loader
        self._extractor = extractor
        self._stop_event: asyncio.Event | None = None

    # ---------- 单批处理 ----------

    async def run_once(self, now: datetime | None = None) -> int:
        """处理一批 job：requeue 到期 failed → claim → 逐个处理。

        Returns:
            处理的 job 数量。
        """
        now = now or datetime.now(timezone.utc)
        # reaper 职责（19.3 + review #4）：processing 超时（lease）与 failed 到期的
        # 恢复/死信；lease_seconds 由 worker 实例传入（不使用硬编码 300）
        await self._repository.recover_due_jobs(
            now,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )

        claimed = await self._repository.claim_jobs(
            self._worker_id, self._claim_limit, now
        )
        processed = 0
        for job in claimed:
            try:
                await self._process_job(job)
            except JobGenerationMismatch:
                await self._cancel_job(job)
            except JobLeaseLost:
                # lease 已回收、其他 worker 接手：本 worker 不再处理，不进入 retry/dead
                logger.warning("event=memory_job_lease_lost outcome=skipped job_id=%s", job.job_id)
            except Exception as exc:
                await self._handle_failure(job, exc)
            processed += 1
        return processed

    # ---------- 常驻循环 ----------

    async def run_forever(self, stop_event: asyncio.Event | None = None) -> None:
        """轮询循环：默认 1 秒，空闲时最多退避到 5 秒（方案 19.3）。"""
        self._stop_event = stop_event or asyncio.Event()
        interval = self._poll_interval
        while not self._stop_event.is_set():
            try:
                processed = await self.run_once()
                interval = (
                    self._poll_interval
                    if processed > 0
                    else min(interval * 2, MAX_POLL_INTERVAL)
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    "event=memory_worker_poll_failed outcome=retry_next_round "
                    "error_type=%s",
                    type(exc).__name__,
                )
                interval = min(interval * 2, MAX_POLL_INTERVAL)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass

    # ---------- job 处理（方案 19.3 固定事务协议） ----------

    async def _process_job(self, job: MemoryJob) -> None:
        """方案 19.3 固定事务协议编排。

        extract_memory：prepare 候选命令（Service/Policy 层）→ apply_extract_job
        （Repository 单事务：generation fence + lease ownership + 写入 + job 终态）。
        expire_memory：generation 快速检查 + 清理 + complete。
        """
        payload = job.payload or {}
        expected_generation = int(payload.get("memory_generation", 0))

        # 快速失败防线（事务内由 apply_extract_job 复核，review #1）
        current_generation = await self._repository.get_memory_generation(job.user_id)
        if current_generation != expected_generation:
            raise JobGenerationMismatch(
                f"generation 不匹配：job={expected_generation} current={current_generation}"
            )

        # 第一版自动路径（方案 6.3/6.4）：确定性工具事件 → tool_verified 候选；
        # 模型抽取候选（model_inferred）一律丢弃，不入库
        if job.job_type == "extract_memory":
            await self._handle_extract(job)
        elif job.job_type == "expire_memory":
            await self._handle_expire(job)

    async def _handle_extract(self, job: MemoryJob) -> None:
        """extract_memory（review #2/#3/#1）：

        1. 依赖缺失 → raise（走 retry/dead，不伪造 succeeded）
        2. 按 payload 的 user_message_id/assistant_message_id 定位本轮区间，
           只取区间内受控工具结果（不扫描整个历史 thread）
        3. Service.prepare_memory_command（Policy/TTL/fingerprint 仍在 Service 层）
        4. Repository.apply_extract_job 单事务提交（generation fence + lease ownership）

        模型 structured extraction 第一版不调用（候选均为 model_inferred，按 6.4 丢弃）；
        extractor 实例注入后仅用于记录指标，不持久化其结果。
        """
        if self._display_messages_loader is None or self._memory_service is None:
            raise RuntimeError(
                "extract worker 缺少 display_messages_loader/memory_service，"
                "不能把 Job 标记 succeeded"
            )

        payload = job.payload or {}
        user_message_id = payload.get("user_message_id")
        assistant_message_id = payload.get("assistant_message_id")
        if not (user_message_id and assistant_message_id):
            raise ReferencedMessageUnavailable("extract job payload 缺少消息 ID 引用")

        loaded = self._display_messages_loader(job.thread_id)
        if inspect.isawaitable(loaded):
            loaded = await loaded
        messages = loaded or []

        # 本轮区间定位（review #3）：user_message_id → assistant_message_id
        user_index = next(
            (i for i, m in enumerate(messages)
             if m.get("id") == user_message_id and m.get("role") == "user"),
            None,
        )
        assistant_index = next(
            (i for i, m in enumerate(messages)
             if m.get("id") == assistant_message_id and m.get("role") == "assistant"),
            None,
        )
        if user_index is None or assistant_index is None:
            raise ReferencedMessageUnavailable(
                f"引用消息缺失（user={user_message_id} assistant={assistant_message_id}），"
                "按方案 19.4 重试，超过消息保留窗口后转 dead"
            )
        if user_index >= assistant_index:
            raise ReferencedMessageUnavailable("引用消息顺序异常（user 在 assistant 之后）")
        round_messages = messages[user_index:assistant_index + 1]

        # 本轮受控工具结果摘要（方案 19.4：只读受控工具结果，不复制完整输出）
        # review 第三轮（producer/consumer 契约）：display_messages 的 tool 消息
        # 结果存 text 字段（chat.py 写入 serialized["text"]），summary 必须读 text；
        # succeeded 由 chat.py 在 ToolMessage 处生成（结构化状态），透传即可
        tool_results = [
            {
                "tool_name": m.get("tool_name"),
                "summary": str(m.get("text") or m.get("content") or "")[:500],
                "succeeded": m.get("succeeded"),
            }
            for m in round_messages
            if m.get("role") == "tool" and m.get("tool_name")
        ]
        candidates = tool_event_candidates(tool_results)

        # 模型抽取（第一版不持久化其结果；存在 extractor 时仅记录指标）
        if self._extractor is not None:
            user_msg = next(
                (m.get("content", "") for m in round_messages
                 if m.get("role") == "user"),
                "",
            )
            assistant_msg = next(
                (m.get("content", "") for m in round_messages
                 if m.get("role") == "assistant"),
                "",
            )
            try:
                result = await self._extractor.extract(
                    user_message=user_msg or "",
                    assistant_message=assistant_msg or "",
                )
                dropped = len(result.memory_candidates)
                if dropped:
                    logger.info(
                        "MemoryWorker: 自动路径模型候选 %s 条全部丢弃（model_inferred 第一版不入库，6.4）",
                        dropped,
                    )
            except Exception as exc:
                # 模型抽取失败不阻断工具事件候选入库；只记录安全摘要（方案 20.3）
                logger.error(
                    "event=memory_candidate_dropped reason_code=model_extract_failed "
                    "outcome=skipped error_type=%s job_id=%s",
                    type(exc).__name__, job.job_id,
                )

        # Service 层准备命令（Policy/敏感/TTL/fingerprint 校验，review #1.3）
        actor = ActorContext(actor_type="system", source_thread_id=job.thread_id)
        commands = [
            self._memory_service.prepare_memory_command(
                user_id=job.user_id,
                actor=actor,
                kind=candidate.kind,
                content=candidate.content,
                data=candidate.data,
                entity_type=candidate.entity_type,
                entity_id=candidate.entity_id,
                source_type=SourceType.TOOL_VERIFIED,
                source_message_id=assistant_message_id,
                long_term=False,
            )
            for candidate in candidates
        ]

        # Repository 单事务原子提交（generation fence + lease ownership，review #1）
        outcome = await self._repository.apply_extract_job(
            job_id=job.job_id,
            worker_id=self._worker_id,
            user_id=job.user_id,
            expected_generation=int(payload.get("memory_generation", 0)),
            commands=commands,
            actor=actor,
        )
        if outcome == "stale":
            # lease 已被 reaper 回收、job 由其他 worker 接手：本 worker 不再处理
            raise JobLeaseLost(f"job {job.job_id} 的 lease 已失效（stale）")
        if outcome == "cancelled":
            logger.warning(
                "event=memory_job_cancelled reason_code=generation_mismatch "
                "outcome=cancelled job_id=%s",
                job.job_id,
            )

    async def _handle_expire(self, job: MemoryJob) -> None:
        """expire_memory：apply_expire_job 单事务（generation fence + lease ownership，
        review 第二轮）——过期 Item 清理、revision 递增、job 终态同事务完成。

        第一版：将过期 Item 标 forgotten（30 天宽限期后物理删除留清理 worker，18.4）。
        """
        payload = job.payload or {}
        cutoff = payload.get("cutoff_at")
        if not cutoff:
            raise RuntimeError("expire_memory job payload 缺少 cutoff_at")

        outcome = await self._repository.apply_expire_job(
            job_id=job.job_id,
            worker_id=self._worker_id,
            user_id=job.user_id,
            expected_generation=int(payload.get("memory_generation", 0)),
            cutoff=cutoff,
        )
        if outcome == "stale":
            raise JobLeaseLost(f"job {job.job_id} 的 lease 已失效（stale）")
        if outcome == "cancelled":
            logger.warning(
                "MemoryWorker: job %s cancelled（generation 不匹配，delete-all 后旧 job 不得修改记忆）",
                job.job_id,
            )

    async def _cancel_job(self, job: MemoryJob) -> None:
        """generation 不匹配：worker 只能将 processing Job 转 cancelled 并清空锁
        （19.3/12.1：不能写 Profile/Item/Event、不能重试、不能进入 dead）。"""
        await self._repository.cancel_job(job.job_id, self._worker_id)
        logger.warning(
            "MemoryWorker: job %s cancelled（generation 不匹配，delete-all 后旧 job 不得复活记忆）",
            job.job_id,
        )

    async def _handle_failure(self, job: MemoryJob, exc: Exception) -> None:
        """失败：指数退避重试；超过最大尝试由 reaper 转 dead。

        last_error 只保存安全摘要（reason code + 类型名，方案 20.3），
        不保存 str(exc)（SQLAlchemy/Pydantic 异常可能携带记忆正文或参数）。
        """
        safe_error = safe_error_summary(exc)
        if job.attempts >= self._max_attempts:
            # 直接转 dead（保留安全摘要供人工排查，方案 19.3）
            await self._repository.dead_job(job.job_id, self._worker_id, safe_error)
            logger.error(
                "event=memory_job_dead reason_code=max_attempts_exceeded "
                "outcome=dead job_id=%s attempts=%s %s",
                job.job_id, self._max_attempts, safe_error,
            )
            return
        retry_at = datetime.now(timezone.utc) + timedelta(seconds=_backoff_seconds(job.attempts))
        await self._repository.fail_job(
            job.job_id, self._worker_id, safe_error, retry_at
        )
        logger.warning(
            "event=memory_job_retried outcome=retry_scheduled "
            "job_id=%s attempts=%s retry_at=%s %s",
            job.job_id, job.attempts, retry_at.isoformat(), safe_error,
        )


# ============================================================
# worker 模式（方案 19.3：MEMORY_WORKER_MODE=disabled|embedded|standalone）
# ============================================================

WORKER_MODE = os.getenv("MEMORY_WORKER_MODE", "disabled")


def worker_mode() -> str:
    if WORKER_MODE not in ("disabled", "embedded", "standalone"):
        raise RuntimeError(
            f"MEMORY_WORKER_MODE 非法值 {WORKER_MODE!r}：只允许 disabled/embedded/standalone"
        )
    return WORKER_MODE
