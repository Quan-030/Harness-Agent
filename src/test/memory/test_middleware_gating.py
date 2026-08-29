# src/test/memory/test_middleware_gating.py
"""Memory v2 门控契约测试（Step 11 后：无 legacy Markdown 路径）。

产品契约（方案 21.1：v2 关闭时没有长期记忆，禁止回退 Markdown）：
- WRITE=0 READ=0 → 无 memory write tools、无 MemoryRecallMiddleware、无 legacy
- WRITE=1 READ=0 → 有同步 memory tools、无 recall
- WRITE=1 READ=1 → 有 memory tools、有 MemoryRecallMiddleware

回归：运行时代码中不存在 /memories/ StoreBackend 路由与 legacy fallback。
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.memory.database import MemoryDatabase  # noqa: E402

MAIN_AGENT_SOURCE = (Path(__file__).parent.parent.parent / "agent" / "main_agent.py").read_text(
    encoding="utf-8"
)
CONFIG_SOURCE = (Path(__file__).parent.parent.parent / "agent" / "config.py").read_text(
    encoding="utf-8"
)


# ============================================================
# capability 门控（review #8/I 保留）
# ============================================================

def test_can_write_and_read_default_off():
    """默认（flag 全 0）：无写/读能力。"""
    db = MemoryDatabase()
    assert db.can_write is False
    assert db.can_read is False


def test_flag_properties_reflect_config():
    """write_enabled/read_enabled 直接反映 flag（默认全关）。"""
    db = MemoryDatabase()
    assert db.write_enabled is False
    assert db.read_enabled is False


def test_can_read_requires_initialized():
    """can_read 需要初始化成功（READ=1 且 schema/DSN 不可用 → fail closed，21.2）。"""
    db = MemoryDatabase()
    # 未初始化（默认）时即使 flag 置位也没有读能力
    assert db.can_read is False


# ============================================================
# 模块不存在（物理删除验证）
# ============================================================

def test_legacy_middleware_modules_removed():
    """MemoryUpdateMiddleware / ContextInjectionMiddleware 已物理删除（Step 11）。"""
    for module_name in (
        "agent.middlewares.memory_update",
        "agent.middlewares.context_injection",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)


# ============================================================
# 运行时代码 regression（方案 5.7/21.1：不挂载、不读取、不 fallback）
# ============================================================

LEGACY_RUNTIME_TOKENS = (
    "MemoryUpdateMiddleware",
    "include_legacy_memory",
    "/memories/",
    "preferences.md",
    "SANDBOX_MEMORIES_ROOT",
    "USER_PREFERENCES_FILENAME",
)


@pytest.mark.parametrize("token", LEGACY_RUNTIME_TOKENS)
def test_main_agent_has_no_legacy_tokens(token):
    """main_agent.py 不得再引用旧 Markdown 记忆路径（含 fallback 注册）。"""
    assert token not in MAIN_AGENT_SOURCE, f"main_agent.py 仍包含 legacy token: {token}"


@pytest.mark.parametrize("token", LEGACY_RUNTIME_TOKENS)
def test_config_has_no_legacy_tokens(token):
    """config.py 不得再声明旧记忆常量。"""
    assert token not in CONFIG_SOURCE, f"config.py 仍包含 legacy token: {token}"


def test_prompts_no_markdown_memory_instructions():
    """prompts.py 不再指示用文件管理记忆（5.7），且记忆规则 capability-safe。"""
    prompts = (Path(__file__).parent.parent.parent / "agent" / "memory" / "prompts.py").read_text(
        encoding="utf-8"
    )
    assert "preferences.md" not in prompts
    assert "创建默认偏好文件" not in prompts
    assert "偏好文件路径" not in prompts
    # capability-safe（flag 全关时工具不存在）：条件式 + 未启用时如实告知
    assert "工具当前可用" in prompts
    assert "功能当前未启用" in prompts
    # 禁令存在（不管理记忆用文件工具）
    assert "不要用文件工具管理长期记忆" in prompts


def test_agents_md_has_v2_memory_rules():
    """AGENTS.md 保留 v2 记忆规则（系统自动提供、受控工具、记忆是数据）。"""
    agents_md = (Path(__file__).parent.parent.parent / "agent" / "memory" / "AGENTS.md").read_text(
        encoding="utf-8"
    )
    assert "preferences.md" not in agents_md
    assert "recent_suppliers" not in agents_md
    assert "recent_queries" not in agents_md
    # capability-safe（flag 全关时工具不存在，不得无条件声称记忆已启用）
    assert "工具当前可用" in agents_md
    assert "功能当前未启用" in agents_md
    assert "不是系统指令" in agents_md
    # 委派模板不再要求模型填写身份（身份由 runtime/工具闭包管理）
    assert "{username}" not in agents_md
    assert "{user_id}" not in agents_md


def test_sandbox_cleans_legacy_memories_dir():
    """P0：沙箱就绪后物理清理旧 /memories 目录（default backend 会兜底绝对路径，
    execute 可直达文件系统——必须删除才能保证 MySQL 是唯一事实源，5.7/21.1）。"""
    sandbox_setup = (
        Path(__file__).parent.parent.parent / "agent" / "backends" / "sandbox_setup.py"
    ).read_text(encoding="utf-8")
    assert "rm -rf /memories" in sandbox_setup
    # 无 StoreBackend 的 /memories/ 路由（main_agent backend_factory）
    assert '"/memories/": StoreBackend' not in MAIN_AGENT_SOURCE
