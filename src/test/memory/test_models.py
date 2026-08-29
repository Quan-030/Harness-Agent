# src/test/memory/test_models.py
"""Memory v2 领域模型契约单元测试（方案 12.1 / 16 节）。

覆盖：严格校验（extra=forbid）、枚举、字符串约束（NUL/空白/长度）、
data discriminated union、entity 成对校验、ProfilePatch 原子操作、
confidence 边界、指纹规范化。
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.memory.models import (
    ChartType,
    ChartTypeOp,
    CurrencyOp,
    DeliveryDaysMaxOp,
    EntityType,
    LanguageOp,
    MemoryCandidate,
    MemoryExtractionResult,
    MemoryItem,
    MemoryKind,
    MemoryProfile,
    MemoryStatus,
    OutputFormat,
    OutputFormatOp,
    PROFILE_SCALAR_OP_ADAPTER,
    ProfileListOp,
    ProfilePatch,
    ProfileScalarOp,
    ProcurementDefaults,
    SourceType,
    compute_fingerprint,
    is_uuid,
    new_uuid7,
    validate_data_by_kind,
)

NOW = datetime(2026, 8, 8, 0, 0, 0, tzinfo=timezone.utc)


# ============================================================
# 通用严格校验
# ============================================================

def test_unknown_fields_rejected():
    """extra=forbid：未知字段一律拒绝。"""
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1",
            version=1,
            created_at=NOW,
            updated_at=NOW,
            unexpected_field="x",
        )
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容",
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
            unexpected_field="x",
        )


def test_invalid_enum_rejected():
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind="bad_kind",  # 非法枚举
            content="内容",
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )


def test_naive_datetime_rejected():
    """时间必须是 UTC-aware，naive datetime 拒绝。"""
    naive = datetime(2026, 8, 8)
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1",
            version=1,
            created_at=naive,
            updated_at=naive,
        )


def test_non_utc_aware_datetime_rejected():
    """review #7：aware 但非 UTC（如 UTC+8）也必须拒绝。"""
    from datetime import timedelta, timezone as tz

    utc8 = tz(timedelta(hours=8))
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1",
            version=1,
            created_at=datetime(2026, 8, 8, 8, 0, 0, tzinfo=utc8),
            updated_at=datetime(2026, 8, 8, 8, 0, 0, tzinfo=utc8),
        )


# ============================================================
# 字符串约束
# ============================================================

def test_nul_character_rejected():
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容\x00恶意",
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )


def test_blank_content_rejected():
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="   \t  ",
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )


def test_content_too_long_rejected():
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="x" * 2001,
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )


def test_user_id_too_long_rejected():
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u" * 256,
            version=1,
            created_at=NOW,
            updated_at=NOW,
        )


# ============================================================
# data discriminated union
# ============================================================

def test_data_unknown_keys_rejected_per_kind():
    """data 必须按 kind 白名单校验，未知键拒绝。"""
    with pytest.raises(Exception):
        validate_data_by_kind(
            MemoryKind.SUPPLIER_CONTEXT,
            {"relationship": "长期合作", "random_key": 1},
        )
    with pytest.raises(Exception):
        validate_data_by_kind(
            MemoryKind.TASK_OUTCOME,
            {"task_type": "询价", "unknown": True},
        )


def test_data_kind_mismatch_rejected():
    """data 结构必须与 kind 匹配。"""
    # supplier_context 不允许 constraint_name（那是 procurement_constraint 的键）
    with pytest.raises(Exception):
        validate_data_by_kind(
            MemoryKind.SUPPLIER_CONTEXT,
            {"constraint_name": "delivery_days_max", "value": 14},
        )


def test_data_union_valid_cases():
    ok = validate_data_by_kind(
        MemoryKind.PROCUREMENT_CONSTRAINT,
        {"constraint_name": "delivery_days_max", "value": 14, "unit": "天"},
    )
    assert ok == {"constraint_name": "delivery_days_max", "value": 14, "unit": "天"}
    ok2 = validate_data_by_kind(
        MemoryKind.USER_FEEDBACK,
        {"target_type": "supplier", "feedback_type": "correction"},
    )
    assert ok2 == {"target_type": "supplier", "feedback_type": "correction"}


def test_entity_type_and_id_must_be_paired():
    """entity_type 与 entity_id 必须同时为空或同时有值。"""
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容",
            entity_type=EntityType.SUPPLIER,
            entity_id=None,
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )
    with pytest.raises(Exception):
        MemoryItem(
            memory_id=new_uuid7(),
            user_id="u1",
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容",
            entity_type=None,
            entity_id="sup-1",
            source_type=SourceType.USER_EXPLICIT,
            source_thread_id="t1",
            fingerprint="f" * 64,
            created_at=NOW,
            updated_at=NOW,
        )


# ============================================================
# MemoryCandidate / ExtractionResult
# ============================================================

def test_candidate_confidence_bounds():
    with pytest.raises(Exception):
        MemoryCandidate(
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容",
            extraction_confidence=1.1,
        )
    with pytest.raises(Exception):
        MemoryCandidate(
            kind=MemoryKind.SUPPLIER_CONTEXT,
            content="内容",
            extraction_confidence=-0.1,
        )


def test_candidate_has_no_source_type_field():
    """MemoryCandidate 不含 source_type（16.3：由调用路径决定来源）。"""
    candidate = MemoryCandidate(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        content="内容",
        extraction_confidence=0.9,
    )
    assert not hasattr(candidate, "source_type")


def test_extraction_result_limits():
    with pytest.raises(Exception):
        MemoryExtractionResult(
            profile_patches=[ProfilePatch(scalar_ops=[{"field": "currency", "op": "set", "value": "CNY"}])] * 21,
        )
    with pytest.raises(Exception):
        MemoryExtractionResult(
            memory_candidates=[
                MemoryCandidate(kind=MemoryKind.TASK_OUTCOME, content=f"c{i}", extraction_confidence=0.9)
                for i in range(21)
            ],
        )


# ============================================================
# Profile / ProfilePatch
# ============================================================

def test_profile_currency_and_language_patterns():
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1", version=1, currency="cny",  # 必须大写 3 位
            created_at=NOW, updated_at=NOW,
        )
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1", version=1, language="x",  # 至少 2 字符
            created_at=NOW, updated_at=NOW,
        )


def test_profile_version_ge_1_and_schema_version_literal():
    with pytest.raises(Exception):
        MemoryProfile(user_id="u1", version=0, created_at=NOW, updated_at=NOW)
    with pytest.raises(Exception):
        MemoryProfile(
            user_id="u1", version=1, schema_version=3,
            created_at=NOW, updated_at=NOW,
        )


def test_profile_roundtrip():
    profile = MemoryProfile(
        user_id="u1",
        output_format=OutputFormat.TABLE,
        chart_type=ChartType.BAR,
        currency="CNY",
        language="zh",
        procurement_defaults=ProcurementDefaults(
            delivery_days_max=14,
            quality_standards=["ISO9001"],
        ),
        version=7,
        created_at=NOW,
        updated_at=NOW,
    )
    assert profile.schema_version == 2
    assert profile.procurement_defaults.delivery_days_max == 14


def _scalar(data: dict) -> object:
    """单独验证标量操作（ProfileScalarOp 是 discriminated union 类型别名，用 TypeAdapter）。"""
    return PROFILE_SCALAR_OP_ADAPTER.validate_python(data)


def test_patch_accepts_output_format_json_value():
    """wire-format 合法枚举字符串 → OutputFormat 枚举实例（主路径 ProfilePatch.model_validate）。"""
    patch = ProfilePatch.model_validate({
        "scalar_ops": [{"field": "output_format", "op": "set", "value": "table"}],
    })
    op = patch.scalar_ops[0]
    assert isinstance(op, OutputFormatOp)
    assert op.value is OutputFormat.TABLE


def test_patch_accepts_chart_type_json_value():
    patch = ProfilePatch.model_validate({
        "scalar_ops": [{"field": "chart_type", "op": "set", "value": "bar"}],
    })
    op = patch.scalar_ops[0]
    assert isinstance(op, ChartTypeOp)
    assert op.value is ChartType.BAR


def test_patch_accepts_non_enum_scalars():
    """非枚举字段：currency/language/delivery_days_max 正常接受。"""
    patch = ProfilePatch.model_validate({
        "scalar_ops": [
            {"field": "currency", "op": "set", "value": "CNY"},
            {"field": "language", "op": "set", "value": "zh-CN"},
            {"field": "delivery_days_max", "op": "set", "value": 14},
        ],
    })
    assert isinstance(patch.scalar_ops[0], CurrencyOp)
    assert isinstance(patch.scalar_ops[1], LanguageOp)
    assert isinstance(patch.scalar_ops[2], DeliveryDaysMaxOp)
    assert patch.scalar_ops[0].value == "CNY"
    assert patch.scalar_ops[2].value == 14


def test_patch_rejects_invalid_enum_json_values():
    """非法枚举值 / 数字 / 枚举串位全部拒绝。"""
    for bad in ("garbage", "scatter", "excel"):
        with pytest.raises(Exception):
            _scalar({"field": "output_format", "op": "set", "value": bad})
    with pytest.raises(Exception):
        _scalar({"field": "chart_type", "op": "set", "value": "scatter"})
    with pytest.raises(Exception):
        _scalar({"field": "output_format", "op": "set", "value": 123})
    with pytest.raises(Exception):
        _scalar({"field": "output_format", "op": "set", "value": "bar"})  # chart 枚举串位


def test_patch_scalar_strict_types_remain_strict():
    """非枚举字段保持 strict：str 不转 int、非 str 拒绝、pattern 生效。"""
    with pytest.raises(Exception):
        _scalar({"field": "delivery_days_max", "op": "set", "value": "14"})
    with pytest.raises(Exception):
        _scalar({"field": "delivery_days_max", "op": "set", "value": 0})
    with pytest.raises(Exception):
        _scalar({"field": "currency", "op": "set", "value": 123})
    with pytest.raises(Exception):
        _scalar({"field": "currency", "op": "set", "value": "cny"})


def test_patch_set_requires_value():
    """set：value 缺失或为 None 都拒绝。"""
    with pytest.raises(Exception):
        _scalar({"field": "currency", "op": "set"})
    with pytest.raises(Exception):
        _scalar({"field": "output_format", "op": "set", "value": None})


def test_patch_clear_semantics():
    """clear：value 省略/None 合法；提供 value 拒绝（fail fast，方案 B'）。"""
    op = _scalar({"field": "currency", "op": "clear"})
    assert op.value is None
    op2 = _scalar({"field": "currency", "op": "clear", "value": None})
    assert op2.value is None
    with pytest.raises(Exception):
        _scalar({"field": "currency", "op": "clear", "value": "CNY"})
    with pytest.raises(Exception):
        _scalar({"field": "output_format", "op": "clear", "value": "table"})


def test_patch_requires_at_least_one_op():
    with pytest.raises(Exception):
        ProfilePatch(scalar_ops=[], list_ops=[])


def test_profile_patch_scalar_ops_schema_is_discriminated():
    """JSON Schema 回归：scalar_ops 是 field-discriminated union（防退化为宽 union）。"""
    schema = ProfilePatch.model_json_schema()
    items = schema["properties"]["scalar_ops"]["items"]
    assert items.get("discriminator", {}).get("propertyName") == "field"
    mapping = items["discriminator"]["mapping"]
    assert set(mapping) == {
        "output_format", "chart_type", "currency", "language", "delivery_days_max",
    }

    def _branch(field_value: str) -> dict:
        ref = mapping[field_value]
        assert ref.startswith("#/$defs/"), f"unexpected ref: {ref}"
        return schema["$defs"][ref.rsplit("/", 1)[-1]]

    def _value_type(field_value: str) -> dict:
        """取可空字段 value schema 的非 null 分支。"""
        value_schema = _branch(field_value)["properties"]["value"]
        if "anyOf" in value_schema:
            return next(
                b for b in value_schema["anyOf"] if b.get("type") != "null"
            )
        return value_schema

    # output_format branch 的 value 是 OutputFormat 枚举（含 text/table/json）
    output_value = _value_type("output_format")
    assert output_value["$ref"] == "#/$defs/OutputFormat"
    assert schema["$defs"]["OutputFormat"]["enum"] == ["text", "table", "json"]
    # delivery_days_max branch 的 value 是 integer
    assert _value_type("delivery_days_max").get("type") == "integer"
    # currency branch 的 value 是 string（pattern）
    currency_value = _value_type("currency")
    assert currency_value.get("type") == "string"
    assert currency_value.get("pattern") == "^[A-Z]{3}$"


def test_patch_list_op_field_length_limits():
    """review #2：Patch 列表元素个数必须不超过目标 Profile 字段上限。"""
    with pytest.raises(Exception):
        ProfileListOp(field="quality_standards", op="replace", values=[f"s{i}" for i in range(51)])
    with pytest.raises(Exception):
        ProfileListOp(field="preferred_regions", op="add", values=[f"r{i}" for i in range(51)])
    # blocked_suppliers 上限 200
    with pytest.raises(Exception):
        ProfileListOp(field="blocked_suppliers", op="replace", values=[f"b{i}" for i in range(201)])
    # 上限边界内合法
    assert len(ProfileListOp(field="quality_standards", op="replace", values=[f"s{i}" for i in range(50)]).values) == 50
    assert len(ProfileListOp(field="blocked_suppliers", op="replace", values=[f"b{i}" for i in range(200)]).values) == 200


def test_list_op_dedupes_keeps_order():
    op = ProfileListOp(field="quality_standards", op="add", values=["A", "B", "a", "A"])
    assert op.values == ["A", "B", "a"]  # 去重保留首次出现顺序


# ============================================================
# 指纹（方案 17.3）
# ============================================================

def _item_data():
    return {
        "kind": MemoryKind.SUPPLIER_CONTEXT,
        "entity_type": EntityType.SUPPLIER,
        "entity_id": "sup-1",
        "content": "  博世  是 长期 合作 伙伴  ",
        "data": {"relationship": "长期合作", "region": "华东"},
    }


def test_fingerprint_normalizes_whitespace_and_case():
    base = compute_fingerprint(**_item_data())
    same = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
        content="博世 是 长期 合作 伙伴",
        data={"region": "华东", "relationship": "长期合作"},  # key 乱序
    )
    # 大小写不同的英文也规范化：BOSCH vs bosch（同一内容应同指纹）
    base_en = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
        content="BOSCH 是 长期 合作 伙伴",
        data={"relationship": "长期合作", "region": "华东"},
    )
    diff_case = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
        content="bosch 是 长期 合作 伙伴",
        data={"relationship": "长期合作", "region": "华东"},
    )
    assert base == same
    assert base_en == diff_case


def test_fingerprint_differs_on_business_values():
    """业务数字与中文标点不得被改写，改变内容必须改变指纹。"""
    a = compute_fingerprint(
        kind=MemoryKind.PROCUREMENT_CONSTRAINT,
        entity_type=None,
        entity_id=None,
        content="交期不超过 14 天",
        data={"constraint_name": "delivery_days_max", "value": 14},
    )
    b = compute_fingerprint(
        kind=MemoryKind.PROCUREMENT_CONSTRAINT,
        entity_type=None,
        entity_id=None,
        content="交期不超过 15 天",  # 数字变化
        data={"constraint_name": "delivery_days_max", "value": 15},
    )
    assert a != b


def test_fingerprint_excludes_user_id():
    """user_id 不进入 hash 内容（17.3），唯一约束靠 (user_id, fingerprint)。"""
    a = compute_fingerprint(**{**_item_data(), "data": {"relationship": "长期合作", "region": "华东"}})
    # 相同业务内容（同一 fingerprint），只是 data 中与 user 无关
    b = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=EntityType.SUPPLIER,
        entity_id="sup-1",
        content="博世 是 长期 合作 伙伴",
        data={"relationship": "长期合作", "region": "华东"},
    )
    assert a == b


def test_fingerprint_data_json_canonical_key_order():
    a = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=None,
        entity_id=None,
        content="内容",
        data={"region": "华东", "relationship": "长期合作"},
    )
    b = compute_fingerprint(
        kind=MemoryKind.SUPPLIER_CONTEXT,
        entity_type=None,
        entity_id=None,
        content="内容",
        data={"relationship": "长期合作", "region": "华东"},
    )
    assert a == b


# ============================================================
# UUIDv7
# ============================================================

def test_uuid7_format_and_uniqueness():
    a, b = new_uuid7(), new_uuid7()
    assert is_uuid(a)
    assert a != b
    # UUIDv7 版本号为 7
    assert a.split("-")[2].startswith("7")
