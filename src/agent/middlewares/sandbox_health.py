"""
沙箱健康检查 + 自动恢复中间件。

在每个 Agent 步骤开始前 ping 沙箱，发现不可用时自动重建沙箱
（skills + venv），并重新上传 AGENTS.md。依赖 SandboxBackendProxy
的热替换能力，重建后其他中间件/工具无需感知变化。

与 SandboxCircuitBreakerMiddleware 配合：
- 本中间件：主动健康检查 → 自动恢复
- SandboxCircuitBreakerMiddleware：恢复后仍失败 → 熔断保护
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from langchain.agents.middleware import AgentMiddleware

from agent.backends.sandbox_proxy import SandboxBackendProxy

logger = logging.getLogger(__name__)


class SandboxHealthMiddleware(AgentMiddleware):
    """沙箱健康守护中间件。

    在 abefore_agent 中 ping 沙箱：
    - 健康 → 透传，零开销
    - 不可用 → 调用 sandbox_manager 重建沙箱 → 重新播种 AGENTS.md → 继续执行
    """

    def __init__(
        self,
        *,
        sandbox_backend: SandboxBackendProxy,
        user_id: str,
        agents_md_content: bytes,
    ) -> None:
        super().__init__()
        self._backend = sandbox_backend
        self._user_id = user_id
        self._agents_md = agents_md_content

    def before_agent(
        self, state: dict[str, Any], runtime: Any
    ) -> Optional[dict[str, Any]]:
        return None

    async def abefore_agent(
        self, state: dict[str, Any], runtime: Any
    ) -> Optional[dict[str, Any]]:
        ok = await self._check()
        if ok:
            return None

        logger.warning("用户 %s 沙箱无响应，触发自动恢复...", self._user_id)
        await self._recover()
        return None

    async def _check(self) -> bool:
        try:
            await asyncio.to_thread(self._backend.execute, "echo ok")
            return True
        except Exception:
            return False

    async def _recover(self) -> None:
        from agent.backends.sandbox_manager import recreate_user_sandbox

        await recreate_user_sandbox(self._user_id)
        await asyncio.to_thread(
            self._backend.upload_files,
            [("/AGENTS.md", self._agents_md)],
        )
        logger.info("用户 %s 沙箱自动恢复完成", self._user_id)
