# src/test/memory/test_policies.py
"""MemoryPolicy / 敏感信息检测单元测试（方案 18.3 / 18.4 / 20.3）。"""
import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.memory.models import MemoryKind  # noqa: E402
from agent.memory.policies import (  # noqa: E402
    EXTRACTION_CONFIDENCE_MIN,
    EXTRACTION_CONFIDENCE_PERSIST,
    MemoryPolicy,
    SensitiveContentError,
    default_ttl,
    detect_sensitive,
)


@pytest.fixture()
def policy():
    return MemoryPolicy()


# ============================================================
# 敏感信息检测（方案 20.3）
# ============================================================

def test_credential_keyword_rejected():
    assert detect_sensitive("我的密码是 abc123", {}) is not None
    assert detect_sensitive("token 值是 xxx", {"access_token": "xxx"}) is not None
    assert detect_sensitive("连接串 mysql://root:pw@1.2.3.4/db", {}) is not None


def test_credit_card_number_rejected():
    assert detect_sensitive("银行卡号 6222021234567890123", {}) is not None


def test_id_card_number_rejected():
    assert detect_sensitive("身份证 11010119900307871X", {}) is not None


def test_credential_uri_rejected():
    assert detect_sensitive("请连接 mysql://user:pass@host:3306/db", {}) is not None
    assert detect_sensitive("使用 postgres://admin:s3cret@db.internal/main", {}) is not None


def test_normal_business_content_allowed():
    """正常采购业务内容不误报。"""
    assert detect_sensitive("用户要求刹车片交期不超过 14 天，金额 5000 元", {}) is None
    assert detect_sensitive("博世是长期合作供应商，质量好", {"relationship": "长期合作"}) is None
    assert detect_sensitive("上次询价订单 PO-20260801 结果成功", {}) is None


def test_detect_does_not_leak_content():
    """拒绝原因只含原因码，不含正文。"""
    reason = detect_sensitive("我的密码是超级机密", {})
    assert reason is not None
    assert "超级机密" not in reason


def test_validate_for_storage_rejects_sensitive(policy):
    with pytest.raises(SensitiveContentError):
        policy.validate_for_storage("access_token 是 xyz", {}, MemoryKind.USER_FEEDBACK)


def test_validate_for_storage_returns_ttl(policy):
    ttl = policy.validate_for_storage("正常内容", {}, MemoryKind.SUPPLIER_CONTEXT)
    assert ttl == timedelta(days=180)


# ============================================================
# TTL 默认值（方案 18.4）
# ============================================================

def test_default_ttl_per_kind():
    assert default_ttl(MemoryKind.SUPPLIER_CONTEXT) == timedelta(days=180)
    assert default_ttl(MemoryKind.PROCUREMENT_CONSTRAINT) == timedelta(days=180)
    assert default_ttl(MemoryKind.TASK_OUTCOME) == timedelta(days=90)
    assert default_ttl(MemoryKind.USER_FEEDBACK) == timedelta(days=365)


# ============================================================
# 抽取阈值（方案 18.3）
# ============================================================

def test_extraction_thresholds():
    assert EXTRACTION_CONFIDENCE_PERSIST == 0.85
    assert EXTRACTION_CONFIDENCE_MIN == 0.65
    # [0.65, 0.85) 丢弃；>= 0.85 入库；< 0.65 丢弃
    assert 0.65 <= 0.8 < EXTRACTION_CONFIDENCE_PERSIST
    assert 0.9 >= EXTRACTION_CONFIDENCE_PERSIST
    assert 0.5 < EXTRACTION_CONFIDENCE_MIN


def test_pem_private_key_rejected(policy):
    """PEM 私钥块（RSA/EC/OPENSSH/ENCRYPTED/DSA 变体）一律拒绝（review #5/H）。"""
    pems = [
        "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0B\n-----END PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIK\n-----END EC PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n-----END OPENSSH PRIVATE KEY-----",
        "-----BEGIN ENCRYPTED PRIVATE KEY-----\nMIIFHDBOBgkqhkiG9w0BBQ0w\n-----END ENCRYPTED PRIVATE KEY-----",
        "-----BEGIN DSA PRIVATE KEY-----\nMIIBuwIBAAKBgQ\n-----END DSA PRIVATE KEY-----",
    ]
    for pem in pems:
        assert detect_sensitive(pem, {}) is not None, f"漏检: {pem[:40]}"


def test_card_number_with_separators_rejected():
    """带空格/短横线的银行卡完整号码拒绝（review #5）。"""
    assert detect_sensitive("卡号 6222 0212 3456 7890 123", {}) is not None
    assert detect_sensitive("卡号 6222-0212-3456-7890-123", {}) is not None


def test_order_number_not_rejected():
    """8 位订单号 / 11 位手机号不误报（review H）。"""
    assert detect_sensitive("订单 PO-20260801 已创建", {}) is None
    assert detect_sensitive("联系电话 13812345678", {}) is None
