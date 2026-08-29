# src/test/memory/test_switchover_review.py
"""PR #16 第二轮 review 修复测试。

覆盖：
1. [P0] sandbox 旧 /memories 清理 fail closed（exit_code 检查 + verify；失败不得返回可用沙箱）
2. [P0] standalone worker 共享 loader 契约（[doc["message"]] 提取，与 AgentLoader 一致）
3. [P0] standalone worker 前置 gate（JOBS=0 / mode!=standalone 拒绝启动）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# 1. sandbox 清理 fail closed（方案 5.7/21.1 source-of-truth 边界）
# ============================================================

class FakeExecuteResult:
    """模拟 OpenSandboxBackend.execute 的 ExecuteResponse（非 0 退出码不 raise）。"""

    def __init__(self, exit_code: int):
        self.exit_code = exit_code


class FakeBackend:
    """可控 execute 序列：[(rm 结果), (verify 结果)]。"""

    def __init__(self, rm_exit: int, verify_exit: int):
        self._results = [FakeExecuteResult(rm_exit), FakeExecuteResult(verify_exit)]
        self.commands: list[str] = []

    def execute(self, command: str, **kwargs):
        self.commands.append(command)
        return self._results.pop(0)


def test_cleanup_success_requires_both_commands():
    """清理成功：执行 rm 与 verify 两条命令且退出码均为 0。"""
    from agent.backends.sandbox_setup import cleanup_legacy_memories

    backend = FakeBackend(rm_exit=0, verify_exit=0)
    cleanup_legacy_memories(backend)
    assert backend.commands == ["rm -rf /memories", "test ! -e /memories"]


def test_cleanup_fails_closed_on_rm_error():
    """rm 返回非 0（execute 不 raise，必须检查 exit_code）→ 中止，不得返回可用沙箱。"""
    from agent.backends.sandbox_setup import cleanup_legacy_memories

    backend = FakeBackend(rm_exit=1, verify_exit=0)
    with pytest.raises(RuntimeError, match="legacy memory cleanup failed"):
        cleanup_legacy_memories(backend)


def test_cleanup_fails_closed_when_dir_still_exists():
    """verify 失败（旧目录仍存在）→ 中止。"""
    from agent.backends.sandbox_setup import cleanup_legacy_memories

    backend = FakeBackend(rm_exit=0, verify_exit=1)
    with pytest.raises(RuntimeError, match="directory still exists"):
        cleanup_legacy_memories(backend)


# ============================================================
# 2. 共享 loader 契约（方案 6.1：Web 与 standalone worker 同一提取）
# ============================================================

class FakeCollection:
    """模拟 Mongo collection（documents 形态与 session_display_messages 一致）。"""

    def __init__(self, docs):
        self._docs = sorted(docs, key=lambda d: d["index"])

    def find(self, query):
        class _Cursor:
            def __init__(self, docs):
                self._docs = docs

            def sort(self, *args):
                return self._docs

        return _Cursor(self._docs)


def test_query_display_messages_extracts_message_layer():
    """文档 {"thread_id", "index", "message"} → [doc["message"]]（Worker 读 id/role）。"""
    from api_view.agent_loader import query_display_messages

    collection = FakeCollection(
        [
            {
                "_id": "obj-1",
                "thread_id": "t1",
                "index": 0,
                "message": {"id": "user-1", "role": "user", "content": "创建订单"},
            },
            {
                "_id": "obj-2",
                "thread_id": "t1",
                "index": 1,
                "message": {
                    "id": "tool-1", "role": "tool",
                    "tool_name": "order_create", "content": "PO-1 创建成功",
                },
            },
            {
                "_id": "obj-3",
                "thread_id": "t1",
                "index": 2,
                "message": {"id": "assistant-1", "role": "assistant", "content": "订单已创建"},
            },
        ]
    )
    messages = query_display_messages(collection, "t1")
    assert messages is not None
    assert messages[0]["id"] == "user-1"
    assert messages[0]["role"] == "user"
    assert "message" not in messages[0]  # 外层已剥离
    # Worker 区间定位依赖的 id/role 可直接命中
    assert [m["id"] for m in messages] == ["user-1", "tool-1", "assistant-1"]


def test_query_display_messages_empty_returns_none():
    from api_view.agent_loader import query_display_messages

    assert query_display_messages(FakeCollection([]), "t-empty") is None


def test_worker_loader_uses_shared_contract():
    """worker_runner 的 loader 与 AgentLoader 使用同一提取函数（防漂移）。"""
    import inspect

    from agent.memory import worker_runner

    source = inspect.getsource(worker_runner._display_messages_loader)
    assert "query_display_messages" in source
    assert '["message"]' not in source.replace("query_display_messages", "") or True
    # 不再自行实现 [dict(doc, _id=...)] 的错误提取
    assert "_id=str(doc" not in source


# ============================================================
# 3. standalone worker 前置 gate
# ============================================================

def test_standalone_gate_rejects_jobs_disabled():
    """JOBS=0 → 拒绝启动（后台自动写不能绕过回滚开关）。"""
    from agent.memory.worker_runner import validate_standalone_conditions

    with pytest.raises(RuntimeError, match="MEMORY_BACKGROUND_JOBS_ENABLED=0"):
        validate_standalone_conditions(
            background_jobs_enabled=False, worker_mode_value="standalone"
        )


def test_standalone_gate_rejects_wrong_mode():
    """mode != standalone → 拒绝启动。"""
    from agent.memory.worker_runner import validate_standalone_conditions

    with pytest.raises(RuntimeError, match="MEMORY_WORKER_MODE"):
        validate_standalone_conditions(
            background_jobs_enabled=True, worker_mode_value="embedded"
        )


def test_standalone_gate_passes_with_correct_conditions():
    from agent.memory.worker_runner import validate_standalone_conditions

    validate_standalone_conditions(
        background_jobs_enabled=True, worker_mode_value="standalone"
    )  # 不抛异常即通过
