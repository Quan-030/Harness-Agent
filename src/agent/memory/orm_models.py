# src/agent/memory/orm_models.py
"""Memory v2 MySQL ORM 模型（SQLAlchemy 2.x async）。

表结构严格对应方案 17.1 的精确持久化契约：
- 所有表使用 InnoDB，字符串默认 utf8mb4，UUID 列显式使用 ASCII
- 枚举使用字符串列 + CHECK 约束（避免原生 ENUM 难以演进）
- 应用时间使用 DATETIME(6)，应用层负责 UTC 转换
- JSON 列写入空对象而非 NULL
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    CHAR,
    JSON,
    CheckConstraint,
    Computed,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, INTEGER, SMALLINT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class MemoryBase(DeclarativeBase):
    """Memory v2 表基类。"""


# 方案 17.1：枚举在 ORM 中使用字符串列，由 Pydantic 与数据库 CHECK 双重限制

# UUID 列：CHAR(36) CHARACTER SET ascii COLLATE ascii_bin（review #4：对齐方案 CHAR 契约）
UUID_COL = CHAR(36, collation="ascii_bin")


class MemoryProfileORM(MemoryBase):
    __tablename__ = "memory_profiles"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    schema_version: Mapped[int] = mapped_column(
        SMALLINT(unsigned=True), nullable=False, server_default=text("2")
    )
    version: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), nullable=False, server_default=text("1")
    )
    output_format: Mapped[str | None] = mapped_column(String(32))
    chart_type: Mapped[str | None] = mapped_column(String(32))
    currency: Mapped[str | None] = mapped_column(CHAR(3))
    language: Mapped[str | None] = mapped_column(String(35))
    procurement_defaults: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "schema_version >= 2",
            name="ck_memory_profiles_schema_version",
        ),
        # review #3：Profile 枚举同样由数据库 CHECK 限制（方案 17.1 双层约束）
        CheckConstraint(
            "output_format IS NULL OR output_format IN ('text','table','json')",
            name="ck_memory_profiles_output_format",
        ),
        CheckConstraint(
            "chart_type IS NULL OR chart_type IN ('bar','line','pie','none')",
            name="ck_memory_profiles_chart_type",
        ),
    )


class MemoryItemORM(MemoryBase):
    __tablename__ = "memory_items"

    memory_id: Mapped[str] = mapped_column(UUID_COL, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(32), nullable=False)
    source_thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    source_message_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'active'")
    )
    fingerprint: Mapped[str] = mapped_column(CHAR(64, collation="ascii_bin"), nullable=False)
    active_fingerprint: Mapped[str | None] = mapped_column(
        CHAR(64, collation="ascii_bin"),
        Computed(
            "CASE WHEN status='active' THEN fingerprint ELSE NULL END",
            persisted=True,
        ),
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6))

    __table_args__ = (
        CheckConstraint(
            "kind IN ('supplier_context','procurement_constraint','task_outcome','user_feedback')",
            name="ck_memory_items_kind",
        ),
        CheckConstraint(
            "source_type IN ('user_explicit','tool_verified','model_inferred')",
            name="ck_memory_items_source_type",
        ),
        CheckConstraint(
            "status IN ('active','superseded','forgotten')",
            name="ck_memory_items_status",
        ),
        CheckConstraint(
            "entity_type IS NULL OR entity_type IN ('supplier','material','order')",
            name="ck_memory_items_entity_type",
        ),
        CheckConstraint(
            "CHAR_LENGTH(content) BETWEEN 1 AND 2000",
            name="ck_memory_items_content_length",
        ),
        CheckConstraint(
            "expires_at IS NULL OR expires_at >= created_at",
            name="ck_memory_items_time_order",
        ),
        UniqueConstraint(
            "user_id", "active_fingerprint",
            name="uq_memory_items_user_active_fingerprint",
        ),
        Index(
            "ix_memory_items_user_status_kind_updated",
            "user_id", "status", "kind", "updated_at",
        ),
        Index(
            "ix_memory_items_user_entity",
            "user_id", "entity_type", "entity_id", "status",
        ),
        Index("ix_memory_items_expires_status", "expires_at", "status"),
    )


class MemoryEventORM(MemoryBase):
    __tablename__ = "memory_events"

    event_id: Mapped[str] = mapped_column(UUID_COL, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    memory_id: Mapped[str | None] = mapped_column(UUID_COL)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    related_memory_id: Mapped[str | None] = mapped_column(UUID_COL)
    source_thread_id: Mapped[str | None] = mapped_column(String(255))
    actor_type: Mapped[str] = mapped_column(String(32), nullable=False)
    reason: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "event_type IN ('created','updated','superseded','forgotten')",
            name="ck_memory_events_event_type",
        ),
        CheckConstraint(
            "actor_type IN ('user','system','admin')",
            name="ck_memory_events_actor_type",
        ),
        Index("ix_memory_events_memory_created", "memory_id", "created_at"),
        Index("ix_memory_events_user_created", "user_id", "created_at"),
    )


class MemoryJobORM(MemoryBase):
    __tablename__ = "memory_jobs"

    job_id: Mapped[str] = mapped_column(UUID_COL, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(
        String(255, collation="ascii_bin"), nullable=False, unique=True
    )
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default=text("'pending'")
    )
    attempts: Mapped[int] = mapped_column(
        INTEGER(unsigned=True), nullable=False, server_default=text("0")
    )
    available_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    locked_at: Mapped[datetime | None] = mapped_column(DATETIME(fsp=6))
    locked_by: Mapped[str | None] = mapped_column(String(255))
    last_error: Mapped[str | None] = mapped_column(String(2000))
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "job_type IN ('extract_memory','expire_memory')",
            name="ck_memory_jobs_job_type",
        ),
        CheckConstraint(
            "status IN ('pending','processing','succeeded','failed','dead','cancelled')",
            name="ck_memory_jobs_status",
        ),
        Index("ix_memory_jobs_status_available_locked", "status", "available_at", "locked_at"),
    )


class MemoryUserStateORM(MemoryBase):
    __tablename__ = "memory_user_state"

    user_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    memory_revision: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), nullable=False, server_default=text("0")
    )
    memory_generation: Mapped[int] = mapped_column(
        BIGINT(unsigned=True), nullable=False, server_default=text("0")
    )
    created_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DATETIME(fsp=6), nullable=False)
