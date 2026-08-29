# src/middlewares/user_skills_restore.py
"""
技能恢复中间件

在每个 Agent 运行周期开始前，将 StoreBackend 中持久化的技能
恢复到沙箱 /skills/{scope}/{skill_name}/ 路径下，使子 Agent 可以通过
渐进式披露发现和使用。

与 SkillsSyncMiddleware 分工：
  - SkillsSyncMiddleware: 本地 src/skills/ → 沙箱（预置技能）
  - UserSkillsRestoreMiddleware: StoreBackend → 沙箱（持久化技能）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware

from agent.config import USER_SKILLS_STORE_BASE_NAMESPACE

logger = logging.getLogger(__name__)


class UserSkillsRestoreMiddleware(AgentMiddleware):
    """从 StoreBackend 恢复持久化技能到沙箱的中间件。"""

    def __init__(self, backend, system_namespace, user_id) -> None:
        """从 StoreBackend 双命名空间恢复持久化技能。

        Args:
            backend: OpenSandboxBackend 实例，负责文件上传。
            system_namespace: 系统技能命名空间元组（共享，所有用户可读）。
            user_id: 用户标识符，用于构建用户隔离命名空间。
        """
        super().__init__()
        self.backend = backend
        self.system_namespace = system_namespace
        self.user_namespace = USER_SKILLS_STORE_BASE_NAMESPACE + (user_id,)

    async def abefore_agent(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """运行前：从双命名空间读取持久化技能，上传到沙箱。

        系统技能先恢复（低优先级），用户技能后恢复（高优先级，可覆盖）。
        """
        store = runtime.store
        # 1. 恢复系统技能（先写入，低优先级）
        sys_files = await self._collect_skills(store, self.system_namespace)
        # 2. 恢复用户技能（后写入，可覆盖同名系统技能）
        usr_files = await self._collect_skills(store, self.user_namespace)
        all_files = sys_files + usr_files
        if all_files:
            await self.backend.aupload_files(all_files)
        return None

    def before_agent(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[Dict[str, Any]]:
        """同步版本：不执行操作（技能恢复仅支持异步）。"""
        return None

    # --------------------- 内部方法 ---------------------
    async def _collect_skills(self, store, namespace) -> List[Tuple[str, bytes]]:
        """
        从指定 StoreBackend 命名空间收集所有持久化技能文件。

        StoreBackend key 格式: /{scope}/{skill_name}/...
        沙箱目标路径: /skills/{scope}/{skill_name}/...

        Returns:
            (沙箱路径, 文件内容字节) 的列表。
        """
        files: List[Tuple[str, bytes]] = []

        # asearch 默认 limit=10 会截断；MongoDBStore 不支持 offset 分页
        # （非零 offset 抛 NotImplementedError），因此一次性取大上限。
        try:
            items = await store.asearch(namespace, limit=10_000)
        except Exception:
            logger.warning("恢复持久化技能失败：读取 Store 异常", exc_info=True)
            return files

        for item in items:
            key = str(item.key).lstrip("/")

            # key 格式: {scope}/{skill_name}/...
            # 映射到: /skills/{scope}/{skill_name}/...
            parts = key.split("/", 1)
            if len(parts) != 2:
                continue
            scope, rest = parts
            sandbox_path = f"/skills/{scope}/{rest}"

            content = item.value
            if isinstance(content, dict):
                content = content.get("content", "")
            # content 为 list 时拼回完整字符串：StoreBackend 按行存（换行符已剥离），
            # assign_skill 存单元素整文件串——"\n".join 对两种格式都正确。
            if isinstance(content, list):
                content = "\n".join(str(part) for part in content)
            if isinstance(content, str):
                content = content.encode("utf-8")
            if not content:
                continue

            files.append((sandbox_path, content))

        return files
