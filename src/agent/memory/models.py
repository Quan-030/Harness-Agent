# src/agent/memory/models.py
"""Memory v2 领域模型契约（方案第 16 节）。

- 所有输入和领域模型使用 Pydantic v2 严格校验（extra=forbid, strict）。
- LLM/API 不得写入 user_id、ID、状态、来源 ID、版本和时间字段；
  它们由可信 runtime 或应用生成。
- 所有字符串拒绝空白值和 NUL 字符。
- 所有应用时间使用带 UTC 时区的 Python datetime。
- data 字段按 kind 校验为 discriminated union，不允许任意键。
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

STRICT_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    strict=True,
    validate_assignment=True,
    str_strip_whitespace=True,
)

# 通用字符串：非空、拒绝纯空白与 NUL 字符
MemoryString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[^\x00]*$",
    ),
]

# 记忆正文：1~2000 字符
MemoryContent = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=2000,
        pattern=r"^[^\x00]*$",
    ),
]

# 标准 UUID 字符串（数据库使用 CHAR(36)）
UuidString = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$",
    ),
]

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def new_uuid7() -> str:
    """生成标准 UUIDv7 字符串（Python 3.14 原生支持，方案 16.1）。"""
    return str(uuid.uuid7())


def is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


# ============================================================
# 枚举
# ============================================================

class OutputFormat(str, Enum):
    TEXT = "text"
    TABLE = "table"
    JSON = "json"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    NONE = "none"


class MemoryKind(str, Enum):
    SUPPLIER_CONTEXT = "supplier_context"
    PROCUREMENT_CONSTRAINT = "procurement_constraint"
    TASK_OUTCOME = "task_outcome"
    USER_FEEDBACK = "user_feedback"


class EntityType(str, Enum):
    SUPPLIER = "supplier"
    MATERIAL = "material"
    ORDER = "order"


class SourceType(str, Enum):
    USER_EXPLICIT = "user_explicit"
    TOOL_VERIFIED = "tool_verified"
    MODEL_INFERRED = "model_inferred"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


# ============================================================
# Profile
# ============================================================

class ProcurementDefaults(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    delivery_days_max: int | None = Field(default=None, ge=1, le=3650)
    quality_standards: list[MemoryString] = Field(default_factory=list, max_length=50)
    preferred_regions: list[MemoryString] = Field(default_factory=list, max_length=50)
    blocked_suppliers: list[MemoryString] = Field(default_factory=list, max_length=200)


class MemoryProfile(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    schema_version: Literal[2] = 2
    user_id: MemoryString = Field(max_length=255)
    output_format: OutputFormat | None = None
    chart_type: ChartType | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    language: str | None = Field(default=None, min_length=2, max_length=35)
    procurement_defaults: ProcurementDefaults = Field(default_factory=ProcurementDefaults)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def _utc_times(self) -> "MemoryProfile":
        _require_utc_aware(self.created_at, "created_at")
        _require_utc_aware(self.updated_at, "updated_at")
        return self


# ============================================================
# ProfilePatch（显式原子操作，方案 16.2 + review B'：按 field discriminated union）
# ============================================================

ProfileListField = Literal[
    "quality_standards", "preferred_regions", "blocked_suppliers"
]


class _ScalarOpBase(BaseModel):
    """标量操作基类：set 必须提供 value；clear 不得提供 value（fail fast，方案 B'）。"""

    model_config = STRICT_MODEL_CONFIG
    op: Literal["set", "clear"]
    value: Any = None  # 类型占位，子类声明具体类型

    @model_validator(mode="after")
    def _validate_op_value(self) -> "_ScalarOpBase":
        if self.op == "set" and self.value is None:
            raise ValueError("op='set' 时必须提供 value")
        if self.op == "clear" and self.value is not None:
            # 自相矛盾的输入 fail fast，而不是静默忽略（方案 B' 严格语义）
            raise ValueError("op='clear' 时不能提供 value")
        return self


class OutputFormatOp(_ScalarOpBase):
    field: Literal["output_format"]
    # 枚举字段 field-level 关闭 strict：JSON "table" → OutputFormat.TABLE
    #（枚举的 wire representation 合法转换，方案 B'）
    value: OutputFormat | None = Field(default=None, strict=False)


class ChartTypeOp(_ScalarOpBase):
    field: Literal["chart_type"]
    value: ChartType | None = Field(default=None, strict=False)


class CurrencyOp(_ScalarOpBase):
    field: Literal["currency"]
    value: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")


class LanguageOp(_ScalarOpBase):
    field: Literal["language"]
    value: str | None = Field(default=None, min_length=2, max_length=35)


class DeliveryDaysMaxOp(_ScalarOpBase):
    field: Literal["delivery_days_max"]
    value: int | None = Field(default=None, ge=1, le=3650)


# 标量操作联合类型：field 作为 discriminator（方案 B'）。
# 类型别名，不能直接构造/调用 .model_validate()，单独验证用 PROFILE_SCALAR_OP_ADAPTER；
# 业务路径统一走 ProfilePatch.model_validate(...)。
ProfileScalarOp = Annotated[
    Union[
        OutputFormatOp,
        ChartTypeOp,
        CurrencyOp,
        LanguageOp,
        DeliveryDaysMaxOp,
    ],
    Field(discriminator="field"),
]

PROFILE_SCALAR_OP_ADAPTER = TypeAdapter(ProfileScalarOp)


# 列表字段 → 目标 Profile 字段的长度上限（review #2：Patch 沿用 Profile 约束）
_LIST_FIELD_MAX: dict[str, int] = {
    "quality_standards": 50,
    "preferred_regions": 50,
    "blocked_suppliers": 200,
}


class ProfileListOp(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    field: ProfileListField
    op: Literal["replace", "add", "remove"]
    values: list[MemoryString] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def _dedupe_keep_order(self) -> "ProfileListOp":
        """列表操作先去除完全重复值并保持首次出现顺序（方案 16.2）。"""
        seen: set[str] = set()
        deduped: list[str] = []
        for v in self.values:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        object.__setattr__(self, "values", deduped)
        # review #2：元素个数必须不超过目标 Profile 字段自身上限
        if len(deduped) > _LIST_FIELD_MAX[self.field]:
            raise ValueError(
                f"field={self.field} 最多允许 {_LIST_FIELD_MAX[self.field]} 个元素"
                f"（去重后 {len(deduped)} 个）"
            )
        return self


class ProfilePatch(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    scalar_ops: list[ProfileScalarOp] = Field(default_factory=list, max_length=20)
    list_ops: list[ProfileListOp] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _requires_at_least_one_op(self) -> "ProfilePatch":
        if not self.scalar_ops and not self.list_ops:
            raise ValueError("ProfilePatch 至少包含一个操作")
        return self


# ============================================================
# Memory Item data（按 kind 的 discriminated union，方案 16.3）
# ============================================================

class SupplierContextData(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    relationship: MemoryString | None = Field(default=None, max_length=200)
    region: MemoryString | None = Field(default=None, max_length=100)
    note_type: Literal["capability", "risk", "relationship", "other"] = "other"


class ProcurementConstraintData(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    constraint_name: Literal[
        "delivery_days_max", "amount_limit", "currency",
        "quality_standard", "region", "supplier_restriction", "other",
    ]
    value: str | int | float | bool
    unit: MemoryString | None = Field(default=None, max_length=32)


class TaskOutcomeData(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    task_type: MemoryString = Field(max_length=100)
    result_status: Literal["succeeded", "failed", "cancelled", "partial"]
    selected_entity_id: MemoryString | None = Field(default=None, max_length=255)


class UserFeedbackData(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    target_type: Literal["response", "supplier", "material", "order", "workflow"]
    feedback_type: Literal["positive", "negative", "correction", "preference"]
    target_id: MemoryString | None = Field(default=None, max_length=255)


# kind -> 对应 data 模型（data 白名单校验表）
_DATA_MODELS: dict[MemoryKind, type[BaseModel]] = {
    MemoryKind.SUPPLIER_CONTEXT: SupplierContextData,
    MemoryKind.PROCUREMENT_CONSTRAINT: ProcurementConstraintData,
    MemoryKind.TASK_OUTCOME: TaskOutcomeData,
    MemoryKind.USER_FEEDBACK: UserFeedbackData,
}


def validate_data_by_kind(kind: MemoryKind, data: dict[str, Any]) -> dict[str, Any]:
    """按 kind 校验 data 为白名单 discriminated union，拒绝任意键。

    校验通过后返回规范化后的 data dict（供 MySQL JSON 写入）。
    """
    model = _DATA_MODELS[kind]
    validated = model.model_validate(data)
    return validated.model_dump(exclude_none=True)


# ============================================================
# MemoryCandidate / MemoryExtractionResult（方案 4.3）
# ============================================================

class MemoryCandidate(BaseModel):
    """LLM 抽取候选。不包含 source_type/状态/ID/时间——由调用路径决定。"""

    model_config = STRICT_MODEL_CONFIG
    # LLM/JSON 输出模型的枚举字段 field-level strict=False（wire 兼容，方案 B' 先例）
    kind: MemoryKind = Field(strict=False)
    content: MemoryContent
    data: dict[str, Any] = Field(default_factory=dict)
    entity_type: EntityType | None = Field(default=None, strict=False)
    entity_id: MemoryString | None = Field(default=None, max_length=255)
    extraction_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_data_and_entity_pair(self) -> "MemoryCandidate":
        # data 必须按 kind 校验为白名单结构
        validated = validate_data_by_kind(self.kind, self.data)
        object.__setattr__(self, "data", validated)
        # entity_type 与 entity_id 要么同时为空，要么同时有值（方案 16.3）
        if (self.entity_type is None) != (self.entity_id is None):
            raise ValueError("entity_type 与 entity_id 必须同时为空或同时有值")
        return self


class MemoryExtractionResult(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    profile_patches: list[ProfilePatch] = Field(default_factory=list, max_length=20)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=20)


# ============================================================
# MemoryItem（方案 4.3 + 16.3，由应用构造）
# ============================================================

class MemoryItem(BaseModel):
    model_config = STRICT_MODEL_CONFIG
    memory_id: UuidString
    user_id: MemoryString = Field(max_length=255)
    kind: MemoryKind
    content: MemoryContent
    data: dict[str, Any] = Field(default_factory=dict)
    entity_type: EntityType | None = None
    entity_id: MemoryString | None = Field(default=None, max_length=255)
    source_type: SourceType
    source_thread_id: MemoryString = Field(max_length=255)
    source_message_id: MemoryString | None = Field(default=None, max_length=255)
    status: MemoryStatus = MemoryStatus.ACTIVE
    fingerprint: MemoryString = Field(max_length=64)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def _validate(self) -> "MemoryItem":
        validated = validate_data_by_kind(self.kind, self.data)
        object.__setattr__(self, "data", validated)
        if (self.entity_type is None) != (self.entity_id is None):
            raise ValueError("entity_type 与 entity_id 必须同时为空或同时有值")
        _require_utc_aware(self.created_at, "created_at")
        _require_utc_aware(self.updated_at, "updated_at")
        if self.expires_at is not None:
            _require_utc_aware(self.expires_at, "expires_at")
        return self


# ============================================================
# 指纹（方案 17.3）
# ============================================================

def _normalize_text(text: str) -> str:
    """规范化：NFKC、去首尾空白、连续空白折叠为一个空格、英文转小写。"""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _canonical_json(data: dict[str, Any]) -> str:
    """UTF-8、key 排序、无多余空格的 canonical serialization。"""
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def compute_fingerprint(
    kind: MemoryKind,
    entity_type: EntityType | None,
    entity_id: str | None,
    content: str,
    data: dict[str, Any],
) -> str:
    """按方案 17.3 计算去重指纹（user_id 不进入 hash 内容）。"""
    fingerprint_input = "\n".join(
        [
            kind.value,
            entity_type.value if entity_type else "",
            entity_id or "",
            _normalize_text(content),
            _canonical_json(validate_data_by_kind(kind, data)),
        ]
    )
    return hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest()


def _require_utc_aware(value: datetime, field: str) -> None:
    """review #7：不仅要求 aware，还要求确实是 UTC（utcoffset() == 0）。"""
    if value.tzinfo is None:
        raise ValueError(f"{field} 必须是带 UTC 时区的 datetime")
    offset = value.utcoffset()
    if offset is None or offset != timedelta(0):
        raise ValueError(
            f"{field} 必须是 UTC 时间（utcoffset()=0），当前 offset: {offset}"
        )


# ============================================================
# Job payload（方案 19.1：每个 job type 独立 payload 且 extra="forbid"）
# ============================================================

class ExtractMemoryPayload(BaseModel):
    """extract_memory job 类型化 payload（消息引用只存在于该 payload 中，19.1）。"""

    model_config = STRICT_MODEL_CONFIG
    schema_version: Literal[1] = 1
    checkpoint_id: MemoryString = Field(max_length=255)
    user_message_id: MemoryString = Field(max_length=255)
    assistant_message_id: MemoryString = Field(max_length=255)
    extractor_version: MemoryString = Field(max_length=64)
    memory_generation: int = Field(ge=0)
    replay_generation: int = Field(ge=0)


class ExpireMemoryPayload(BaseModel):
    """expire_memory job 类型化 payload（不得包含消息 ID，19.1）。"""

    model_config = STRICT_MODEL_CONFIG
    scan_partition: MemoryString = Field(max_length=64)
    cutoff_at: datetime
    policy_version: MemoryString = Field(max_length=64)
    memory_generation: int = Field(ge=0)
    replay_generation: int = Field(ge=0)

    @model_validator(mode="after")
    def _utc_time(self) -> "ExpireMemoryPayload":
        _require_utc_aware(self.cutoff_at, "cutoff_at")
        return self


JOB_PAYLOAD_MODELS: dict[str, type[BaseModel]] = {
    "extract_memory": ExtractMemoryPayload,
    "expire_memory": ExpireMemoryPayload,
}


def validate_job_payload(job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """按 job_type 校验 payload（extra=forbid：拒绝把 extract_memory 消息字段误传其他 job）。"""
    model = JOB_PAYLOAD_MODELS[job_type]
    return model.model_validate(payload).model_dump(mode="json")
