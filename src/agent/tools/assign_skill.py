# src/tools/assign_skill.py
"""
技能分配工具

将 /skills/main/ 下已验证的技能分配给指定 Agent（主 Agent 或子 Agent）。
支持 StoreBackend 持久化 + 压缩包清理。
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from agent.config import LOCAL_SKILLS_DIR, SCOPE_MAP, SYSTEM_SKILLS_STORE_NAMESPACE, USER_SKILLS_STORE_BASE_NAMESPACE


def create_assign_skill_tool(sandbox_backend, store, user_id):
    """
    创建 assign_skill 工具工厂函数。

    Args:
        sandbox_backend: OpenSandboxBackend 实例，用于沙箱内文件操作。
        store: BaseStore 实例（来自 config.STORE），用于持久化技能到 StoreBackend。
        user_id: 用户标识符，技能持久化到用户隔离命名空间 ("user_skills", user_id)。

    Returns:
        assign_skill 工具函数（异步）。
    """
    from langchain_core.tools import tool

    user_skills_namespace = USER_SKILLS_STORE_BASE_NAMESPACE + (user_id,)

    @tool
    async def assign_skill(skill_name: str, agent_name: str) -> str:
        """
        将已验证的技能分配给指定 Agent（主 Agent 或子 Agent），并持久化到长期存储。

        前提条件：技能已下载/创建到 /skills/main/{skill_name}/ 并通过测试。

        Args:
            skill_name: 技能目录名（如 "web-scraper"）
            agent_name: 目标 Agent：
                - "main" — 分配给主 Agent 自身（技能已就位，直接持久化）
                - "procurement-analyst" — 分配给采购分析子 Agent
                - "procurement-order" — 分配给采购订单子 Agent

        Returns:
            分配确认或错误信息。
        """
        # 1. 校验目标 Agent → scope
        if agent_name not in SCOPE_MAP:
            available = ", ".join(SCOPE_MAP.keys())
            return f"错误：未知 Agent '{agent_name}'。可用: {available}"

        scope = SCOPE_MAP[agent_name]
        source_dir = f"/skills/main/{skill_name}"
        target_dir = f"/skills/{scope}/{skill_name}"

        # 2. 检查源技能是否存在
        check = sandbox_backend.execute(f"test -f {source_dir}/SKILL.md")
        if check.exit_code != 0:
            return (
                f"错误：技能 '{skill_name}' 在 {source_dir}/ 下不存在。\n"
                f"请先完成技能下载/创建和测试。"
            )

        # 2.5. 判断是否为系统技能（已存在于 ("system_skills",) 共享命名空间）
        #       系统技能所有用户共享，无需持久化到用户隔离命名空间。
        is_system = await _is_system_skill(store, skill_name)

        # 3. 持久化到 StoreBackend（仅对用户自创技能执行）
        if is_system:
            persist_report = "📋 系统技能，无需持久化（已存在于共享命名空间）"
        else:
            try:
                persist_report = await _persist_skill(
                    sandbox_backend, store, user_skills_namespace, skill_name, scope
                )
            except Exception as e:
                return (
                    f"错误：持久化时发生异常: {e}\n"
                    f"技能文件仍保留在 {source_dir}/，未执行分配。请重试。"
                )
            # 持久化未完全成功时，中止流程，保留源目录以防数据丢失
            if persist_report.startswith("⚠️"):
                return (
                    f"错误：持久化未完全成功，已中止分配以避免数据丢失。\n"
                    f"{persist_report}\n"
                    f"技能文件仍保留在 {source_dir}/。"
                )

        # 4. 复制到目标 scope 目录并清理源目录（主 Agent 已就位，无需操作）
        if agent_name == "main":
            cp_result = "（主 Agent 技能已就位，无需移动）"
        else:
            result = sandbox_backend.execute(
                f"mkdir -p {target_dir} && cp -r {source_dir}/* {target_dir}/"
            )
            if result.exit_code != 0:
                return f"错误：复制技能文件失败:\n{result.output}"
            # 清理源目录，防止主 Agent 通过 /skills/main/ 发现子 Agent 技能
            sandbox_backend.execute(f"rm -rf {source_dir}")
            verify = sandbox_backend.execute(f"ls {target_dir}/")
            cp_result = (
                f"✅ 已复制到沙箱 {target_dir}/\n"
                f"文件:\n{verify.output.strip()}"
            )

        # 5. 清理压缩包（/skills/main/ 下的 *.zip、*.tar.gz、*.tar、*.tgz）
        cleanup_report = _cleanup_packages(sandbox_backend)

        return (
            f"✅ 技能 '{skill_name}' 已分配给 Agent '{agent_name}'（scope: {scope}）\n"
            f"{cp_result}\n"
            f"{persist_report}\n"
            f"{cleanup_report}"
        )

    assign_skill.name = "assign_skill"
    return assign_skill


# ============================================================
# 内部辅助函数
# ============================================================

async def _persist_skill(sandbox_backend, store, namespace, skill_name: str, scope: str) -> str:
    """将技能文件写入 StoreBackend 持久化。

    从沙箱 /skills/main/{skill_name}/ 读取所有文件，
    写入 store namespace 下 key: /{scope}/{skill_name}/...

    Returns:
        持久化结果描述。
    """
    source_dir = f"/skills/main/{skill_name}"
    now = datetime.now(timezone.utc).isoformat()

    # 列出源目录下所有文件
    ls_result = sandbox_backend.execute(f"find {source_dir} -type f")
    if ls_result.exit_code != 0:
        return f"⚠️ 持久化失败：无法列出 {source_dir}/ 下的文件"

    file_paths = [p.strip() for p in ls_result.output.strip().split("\n") if p.strip()]
    if not file_paths:
        return "⚠️ 持久化跳过：源目录为空"

    persisted_count = 0
    for sandbox_path in file_paths:
        # 计算相对路径 → StoreBackend key
        # 例如 /skills/main/web-fetcher/SKILL.md → /main/web-fetcher/SKILL.md
        rel = sandbox_path[len(f"/skills/main/"):]
        store_key = f"/{scope}/{rel}"

        # 读取文件内容
        try:
            dl = sandbox_backend.download_files([sandbox_path])
            if not dl or dl[0].error:
                continue
            content_bytes = dl[0].content
            content_str = content_bytes.decode("utf-8") if isinstance(content_bytes, bytes) else str(content_bytes)
        except Exception:
            continue

        # 写入 Store（格式与 StoreBackend 一致）
        try:
            await store.aput(
                namespace,
                store_key,
                {
                    "content": [content_str],
                    "created_at": now,
                    "modified_at": now,
                },
            )
            persisted_count += 1
        except Exception as e:
            return f"⚠️ 持久化部分失败（{store_key}: {e}），已成功 {persisted_count} 个文件"

    return f"💾 持久化完成：{persisted_count} 个文件 → StoreBackend /persisted-skills/{scope}/{skill_name}/"


def _cleanup_packages(sandbox_backend) -> str:
    """删除 /skills/main/ 下的压缩包文件。

    Returns:
        清理结果描述。
    """
    patterns = "*.zip *.tar.gz *.tar *.tgz *.tar.bz2 *.tar.xz"
    cmd = f"cd /skills/main/ && rm -f {patterns} 2>/dev/null; ls {patterns} 2>/dev/null || echo 'none'"
    result = sandbox_backend.execute(cmd)

    output = result.output.strip()
    if output == "none" or not output:
        return "🧹 压缩包已清理"
    else:
        return f"🧹 压缩包已清理（残留: {output}）"


async def _is_system_skill(store, skill_name: str) -> bool:
    """判断技能是否已存在于系统共享命名空间。

    查询 ("system_skills",) 命名空间中是否存在 key 匹配
    模式 ^/[^/]+/{skill_name}/ 的条目（如 /main/skill-management/SKILL.md）。
    查询失败时回退本地文件系统检查（src/skills/*/{skill_name}/SKILL.md）。
    两层都失败返回 False 兜底，保留旧行为防止数据丢失。

    Args:
        store: BaseStore 实例。
        skill_name: 技能目录名。

    Returns:
        True 表示该技能是系统预置技能，无需持久化到用户命名空间。
    """
    # 第一层：查询 StoreBackend 系统命名空间
    try:
        items = await store.asearch(SYSTEM_SKILLS_STORE_NAMESPACE, limit=10_000)
    except Exception:
        items = None

    if items:
        pattern = re.compile(rf"^/[^/]+/{re.escape(skill_name)}/")
        for item in items:
            if pattern.search(str(item.key)):
                return True
        return False

    # 第二层：Store 不可用，回退本地文件系统
    try:
        local_dir = Path(LOCAL_SKILLS_DIR)
        skill_md = list(local_dir.rglob(f"{skill_name}/SKILL.md"))
        if skill_md:
            return True
    except Exception:
        pass

    # 两层都失败，兜底返回 False
    return False
