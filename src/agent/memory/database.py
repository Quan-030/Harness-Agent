# src/agent/memory/database.py
"""Memory v2 MySQL engine / session 生命周期与健康检查（方案 5.8 / 21.2 / 9 节）。

启动语义（与用户对齐确认）：
- 完全关闭（READ=0 且 WRITE=0 且 JOBS=0 且 SEMANTIC=0）→ 跳过 MySQL 初始化，正常启动
- 启用任意能力 → DSN 必须有效、连接必须成功、migration 版本必须匹配，否则 fail closed
- flag 非法组合（方案 21.2）→ 启动失败
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def validate_memory_flags(
    *,
    write_enabled: bool,
    read_enabled: bool,
    jobs_enabled: bool,
    semantic_enabled: bool,
) -> None:
    """按方案 21.2 校验 feature flag 合法组合，非法组合抛 RuntimeError。"""
    if write_enabled is False and jobs_enabled is True:
        raise RuntimeError(
            "非法 flag 组合：WRITE=0 且 JOBS=1。后台 Job 会产生自动记忆，"
            "不能在写入关闭时启用（方案 21.2）。"
        )
    if write_enabled and not read_enabled and jobs_enabled:
        raise RuntimeError(
            "非法 flag 组合：WRITE=1 READ=0 JOBS=1。后台 Job 会产生用户不可见的自动记忆，"
            "不能作为首次启用状态（方案 21.2）。"
        )
    if semantic_enabled and not read_enabled:
        raise RuntimeError(
            "非法 flag 组合：SEMANTIC_RETRIEVAL=1 且 READ=0。语义检索依赖读取（方案 21.2）。"
        )


def memory_v2_enabled(
    *,
    write_enabled: bool,
    read_enabled: bool,
    jobs_enabled: bool,
    semantic_enabled: bool,
) -> bool:
    """任一 Memory v2 能力启用即需要 MySQL 就绪。"""
    return any(
        [write_enabled, read_enabled, jobs_enabled, semantic_enabled]
    )


async def check_memory_database_health(
    engine: AsyncEngine,
    expected_revision: str,
) -> None:
    """健康检查：连接可达 + alembic migration 版本匹配（方案 5.8）。

    任一不满足抛 RuntimeError，调用方须 fail closed。
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            row = (
                await conn.execute(
                    text("SELECT version_num FROM alembic_version")
                )
            ).first()
    except Exception as exc:  # 连接失败/表不存在等
        raise RuntimeError(
            f"Memory v2 MySQL 健康检查失败：无法读取 alembic_version（{exc}）"
        ) from exc

    if row is None or row.version_num != expected_revision:
        raise RuntimeError(
            f"Memory v2 migration 版本不匹配：期望 {expected_revision}，"
            f"实际 {getattr(row, 'version_num', None)}。"
            "不允许带错误 schema 启动（方案 5.8），请先执行 alembic upgrade head。"
        )


class MemoryDatabase:
    """Memory v2 MySQL 生命周期管理器（FastAPI lifespan 内 initialize/dispose）。"""

    def __init__(self) -> None:
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None
        self._initialized = False
        # capability（review I：initialize 成功后一次性发布，
        # 避免调用方分别 import config 组合条件导致状态不一致）
        self.write_enabled = False
        self.read_enabled = False
        self.jobs_enabled = False
        self.semantic_enabled = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def can_write(self) -> bool:
        """写能力（显式同步写入工具/API 可用性；review #8）。"""
        return self._initialized and self.write_enabled

    @property
    def can_read(self) -> bool:
        """读能力（召回/读取可用性；review #8）。"""
        return self._initialized and self.read_enabled

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("MemoryDatabase 未初始化：请先调用 initialize()")
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession] | None:
        """Repository/Service 依赖注入入口；未初始化时返回 None（API 返回 503）。"""
        return self._session_factory

    async def initialize(
        self,
        *,
        dsn: str | None,
        pool_size: int,
        pool_max_overflow: int,
        connect_timeout: int,
        expected_revision: str,
        write_enabled: bool,
        read_enabled: bool,
        jobs_enabled: bool,
        semantic_enabled: bool,
    ) -> bool:
        """初始化 MySQL 连接池并做健康检查。

        Returns:
            True 表示 Memory v2 已启用且就绪；False 表示完全关闭（跳过初始化）。
        Raises:
            RuntimeError: flag 非法组合 / 启用但 DSN 缺失或健康检查失败（fail closed）。
        """
        validate_memory_flags(
            write_enabled=write_enabled,
            read_enabled=read_enabled,
            jobs_enabled=jobs_enabled,
            semantic_enabled=semantic_enabled,
        )
        if not memory_v2_enabled(
            write_enabled=write_enabled,
            read_enabled=read_enabled,
            jobs_enabled=jobs_enabled,
            semantic_enabled=semantic_enabled,
        ):
            return False

        if not dsn:
            raise RuntimeError(
                "Memory v2 已启用但 MEMORY_MYSQL_DSN 未配置：fail closed（方案 5.8）。"
            )

        # review #5：先用局部变量创建并检查，全部成功后再一次性发布到实例状态，
        # 避免 health check 失败留下半初始化 engine（_engine 非 None 但未 dispose）
        engine = create_async_engine(
            dsn,
            pool_size=pool_size,
            max_overflow=pool_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            # 方案 20.3 第二层保护：SQL 参数（记忆正文/查询词）不出现在异常与日志中
            hide_parameters=True,
            connect_args={
                "connect_timeout": connect_timeout,
                "read_timeout": connect_timeout,
            },
        )
        try:
            await check_memory_database_health(engine, expected_revision)
        except Exception:
            await engine.dispose()
            raise

        self._engine = engine
        self._session_factory = async_sessionmaker(
            engine, expire_on_commit=False
        )
        # 一次性发布 capability（review I）
        self.write_enabled = write_enabled
        self.read_enabled = read_enabled
        self.jobs_enabled = jobs_enabled
        self.semantic_enabled = semantic_enabled
        self._initialized = True
        return True

    async def dispose(self) -> None:
        """关闭连接池（应用 shutdown 时调用）。"""
        if self._engine is not None:
            await self._engine.dispose()
        self._engine = None
        self._session_factory = None
        self.write_enabled = False
        self.read_enabled = False
        self.jobs_enabled = False
        self.semantic_enabled = False
        self._initialized = False

    def session(self) -> AsyncSession:
        """为 Repository/Service 提供会话（未初始化时 fail closed）。"""
        if self._session_factory is None:
            raise RuntimeError("MemoryDatabase 未初始化：无法创建 session")
        return self._session_factory()


memory_database = MemoryDatabase()
