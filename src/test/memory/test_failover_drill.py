# src/test/memory/test_failover_drill.py
"""MySQL 故障降级演练（Step 12，方案 5.8）。

演练内容：
1. MySQL 不可用（DSN 指向无服务端口）→ 启用 READ=1 时启动 fail closed
2. MySQL 不可用时 WRITE=1 预热态同样 fail closed（DSN 不可用即启动失败）
3. 完全关闭（flag 全 0）→ 无 DSN 也能正常跳过（不要求 MySQL）

真实连接失败演练：连接 127.0.0.1:3307（本机未监听）模拟 MySQL 停机。
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.memory.database import MemoryDatabase  # noqa: E402

# 演练 A/B：指向无服务端口（真实 TCP 连接失败，等价 MySQL 停机）
UNREACHABLE_DSN = "mysql+asyncmy://root:wrong@127.0.0.1:3307/memory_v2_drill"

FLAGS = dict(
    write_enabled=False,
    read_enabled=False,
    jobs_enabled=False,
    semantic_enabled=False,
)


def test_drill_a_read_enabled_fails_closed_on_mysql_down():
    """MySQL 停机 + READ=1 → 启动失败（fail closed，不允许无记忆可读运行）。"""
    db = MemoryDatabase()

    async def _run():
        return await db.initialize(
            dsn=UNREACHABLE_DSN,
            pool_size=1,
            pool_max_overflow=1,
            connect_timeout=2,
            expected_revision="0002",
            **dict(FLAGS, read_enabled=True),
        )

    with pytest.raises(Exception):
        asyncio.run(_run())
    assert db.initialized is False


def test_drill_b_write_only_fails_closed_on_mysql_down():
    """MySQL 停机 + WRITE=1 预热态 → 同样启动失败（DSN 不可用即 fail closed）。"""
    db = MemoryDatabase()

    async def _run():
        return await db.initialize(
            dsn=UNREACHABLE_DSN,
            pool_size=1,
            pool_max_overflow=1,
            connect_timeout=2,
            expected_revision="0002",
            **dict(FLAGS, write_enabled=True),
        )

    with pytest.raises(Exception):
        asyncio.run(_run())
    assert db.initialized is False


def test_drill_c_fully_disabled_does_not_require_mysql():
    """MySQL 停机 + 完全关闭 → 跳过初始化，服务正常运行（无长期记忆）。"""
    db = MemoryDatabase()

    async def _run():
        return await db.initialize(
            dsn=None,  # 无 DSN 也应正常跳过
            pool_size=1,
            pool_max_overflow=1,
            connect_timeout=2,
            expected_revision="0002",
            **FLAGS,
        )

    assert asyncio.run(_run()) is False
    assert db.initialized is False
