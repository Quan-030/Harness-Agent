# src/agent/memory/policies.py
"""Memory v2 领域策略（方案 18.1 / 18.3 / 18.4 / 20.3）。

MemoryPolicy 负责：敏感信息检测、TTL 默认值、抽取阈值、来源优先级。
重复与冲突规则由 Repository 承载（17.3），数据白名单由 Pydantic 模型承载（16.3）。
"""
from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

from agent.memory.models import MemoryKind, SourceType

# ============================================================
# 抽取阈值（方案 18.3）
# ============================================================

EXTRACTION_CONFIDENCE_PERSIST = 0.85
""">= 0.85 进入规则校验，通过后允许入库。"""

EXTRACTION_CONFIDENCE_MIN = 0.65
"""< 0.65 直接丢弃；[0.65, 0.85) 后台自动路径一律丢弃，不入库、不跨轮确认。"""

# ============================================================
# 来源优先级（方案 6.4：用户当前明确陈述 > 工具验证结果 > 模型推断）
# ============================================================

SOURCE_PRIORITY: dict[SourceType, int] = {
    SourceType.USER_EXPLICIT: 3,
    SourceType.TOOL_VERIFIED: 2,
    SourceType.MODEL_INFERRED: 1,
}


# ============================================================
# TTL 默认值（方案 18.4）
# ============================================================

DEFAULT_TTL: dict[MemoryKind, timedelta] = {
    MemoryKind.SUPPLIER_CONTEXT: timedelta(days=180),
    MemoryKind.PROCUREMENT_CONSTRAINT: timedelta(days=180),
    MemoryKind.TASK_OUTCOME: timedelta(days=90),
    MemoryKind.USER_FEEDBACK: timedelta(days=365),
}


def default_ttl(kind: MemoryKind) -> timedelta | None:
    """默认 TTL；Profile 稳定偏好不自动过期（返回 None 由调用方处理）。"""
    return DEFAULT_TTL.get(kind)


# ============================================================
# 敏感信息检测（方案 20.3）
# ============================================================

# 凭据类关键词（出现在 content/data 键或文本中即拒绝；含中文常用词）
_CREDENTIAL_KEYWORDS = (
    "password",
    "passwd",
    "pwd",
    "secret",
    "api_key",
    "apikey",
    "api-key",
    "private_key",
    "private-key",
    "private key",  # review #5：空格形式漏检
    "access_token",
    "refresh_token",
    "authorization",
    "bearer ",
    "cookie",
    "dsn",
    "jdbc:",
    "密码",
    "口令",
    "令牌",
    "密钥",
    "卡号",
    "身份证号",
    "护照号",
    "验证码",
)

# 凭据类模式：银行卡完整号码 / 身份证 / 护照 / 带凭据连接串 / PEM 私钥
_CREDENTIAL_PATTERNS = (
    # 银行卡：16-19 位数字（含空格/短横线分隔，review #5/H——先规范化后检查长度）
    re.compile(r"\b\d{16,19}\b"),
    re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?:[ -]?\d{1,3})?\b"),
    re.compile(r"\b\d{17}[\dXx]\b"),  # 18 位身份证号
    re.compile(r"\b[A-Za-z]{2}\d{6,8}\b"),  # 护照号（字母+6-8 位数字）
    re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/@]+:[^\s/@]+@"),  # 带凭据的连接串
    re.compile(r"\b(?:mysql|postgres|mongodb|redis|amqp)://[^\s/@]+:[^\s/@]+@"),
    # PEM 私钥头（review H：不写死算法类型，覆盖 RSA/EC/OPENSSH/ENCRYPTED/DSA 等）
    re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----", re.IGNORECASE),
)


class SensitiveContentError(Exception):
    """记忆内容命中敏感信息禁存规则（方案 20.3：整条拒绝）。"""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(f"内容命中敏感信息禁存规则（{reason_code}）")


def detect_sensitive(content: str, data: dict[str, Any]) -> str | None:
    """检测敏感信息，命中返回拒绝原因码，未命中返回 None。

    命中时整条拒绝，不把被删字段剩余内容拼成可能误导的记忆（方案 20.3）。
    只返回原因码，不返回正文（日志/指标不含敏感正文）。
    """
    text = content
    data_keys = " ".join(str(k).lower() for k in (data or {}).keys())
    data_values = " ".join(str(v) for v in (data or {}).values())
    haystack = f"{text}\n{data_keys}\n{data_values}"

    for keyword in _CREDENTIAL_KEYWORDS:
        if keyword in haystack.lower():
            return f"credential_keyword:{keyword}"

    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(haystack):
            return f"credential_pattern:{pattern.pattern[:40]}"

    return None


class MemoryPolicy:
    """记忆业务策略入口：敏感检测 + TTL + 阈值。

    Repository 负责 SQL 与冲突规则，Policy 不做持久化（方案 18.1 分层）。
    """

    def validate_for_storage(
        self, content: str, data: dict[str, Any], kind: MemoryKind
    ) -> timedelta | None:
        """入库前校验：敏感信息检测 + 返回默认 TTL。

        Raises:
            SensitiveContentError: 命中凭据类禁存规则（管理员/tool_verified 也不能绕过）。
        """
        reason = detect_sensitive(content, data)
        if reason is not None:
            raise SensitiveContentError(reason)
        return default_ttl(kind)
