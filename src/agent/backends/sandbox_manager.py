"""
沙箱管理器 — Per-User 沙箱生命周期管理 + 预热机制。

每个用户维护一个独立的 OpenSandbox，同一个用户的多条对话线程共享。
启动时预热一个沙箱，第一个用户连接时直接认领，无需等待创建。

沙箱 ID 通过 MongoDB 持久化，重启后可重连到已有沙箱。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from pymongo import MongoClient

from agent.backends.sandbox_proxy import SandboxBackendProxy
from agent.config import SANDBOX_CONFIG

logger = logging.getLogger(__name__)

SANDBOX_COLLECTION = "sandbox_registry"

# ---- 全局状态 ----

SANDBOX_BACKENDS: dict[str, SandboxBackendProxy] = {}

_warm_reserve: SandboxBackendProxy | None = None
_warm_lock = asyncio.Lock()

_mongo_client: MongoClient | None = None
_collection = None


def _sandbox_collection():
    if _mongo_client is None:
        raise RuntimeError("沙箱管理器未初始化，请先调用 initialize()")
    from agent.config import MONGODB_DB_NAME
    return _mongo_client[MONGODB_DB_NAME][SANDBOX_COLLECTION]


# ---- 公开 API ----


async def initialize(mongo_client: MongoClient) -> None:
    """启动时初始化 MongoDB 连接。"""
    global _mongo_client, _collection
    _mongo_client = mongo_client
    _collection = _sandbox_collection()
    _collection.create_index("user_id", unique=True)
    logger.info("沙箱管理器已初始化")


async def pre_warm() -> None:
    """启动时预热沙箱，首个用户连接时直接认领。失败不阻塞启动。"""
    global _warm_reserve
    logger.info("正在预热沙箱...")
    try:
        from agent.backends.sandbox_setup import setup_sandbox
        sandbox_backend = await asyncio.to_thread(setup_sandbox, SANDBOX_CONFIG)
        _warm_reserve = SandboxBackendProxy(sandbox_backend)
        logger.info("预热沙箱就绪: %s", sandbox_backend.id)
    except Exception:
        logger.warning("预热沙箱失败，将在第一个用户连接时创建", exc_info=True)
        _warm_reserve = None


async def ensure_sandbox_for_user(user_id: str) -> SandboxBackendProxy:
    """
    获取或创建某个用户的沙箱（四级查找）。

    1. 内存缓存命中 → ping 健康检查 → 失败则重建
    2. MongoDB 有 sandbox_id → connect(id) 重连自己的旧沙箱（优先于预热沙箱，
       保住沙箱内的用户文件）→ 失败则降级到 3
    3. 认领预热沙箱（认领前验活，死沙箱丢弃）
    4. 无货或已死 → 创建新沙箱
    """
    # 状态 1: 内存缓存命中
    proxy = SANDBOX_BACKENDS.get(user_id)
    if proxy is not None:
        try:
            await asyncio.to_thread(proxy.execute, "echo ok")
            logger.info("用户 %s 命中沙箱缓存: %s", user_id, proxy.id)
            return proxy
        except Exception:
            logger.warning("用户 %s 的沙箱 %s 不可达，重建中...", user_id, proxy.id)
            return await _recreate_sandbox(user_id, proxy)

    # 状态 2: MongoDB 重连（先于认领预热沙箱，否则老用户会丢失旧沙箱里的文件）
    doc = _sandbox_collection().find_one({"user_id": user_id})
    if doc and doc.get("sandbox_id"):
        old_id = doc["sandbox_id"]
        logger.info("用户 %s 尝试重连沙箱: %s", user_id, old_id)
        proxy = await _reconnect_sandbox(old_id)
        if proxy is not None:
            SANDBOX_BACKENDS[user_id] = proxy
            _sandbox_collection().update_one(
                {"user_id": user_id},
                {"$set": {"updated_at": datetime.now(timezone.utc)}},
            )
            return proxy
        logger.warning("重连沙箱 %s 失败，为用户 %s 分配新沙箱", old_id, user_id)
        await _try_delete_sandbox(old_id)

    # 状态 3: 认领预热沙箱（含验活）
    proxy = await _claim_warm_reserve()
    if proxy is not None:
        SANDBOX_BACKENDS[user_id] = proxy
        _sandbox_collection().update_one(
            {"user_id": user_id},
            {"$set": {
                "sandbox_id": proxy.id,
                "updated_at": datetime.now(timezone.utc),
                "created_at": datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        logger.info("用户 %s 认领了预热沙箱: %s", user_id, proxy.id)
        return proxy

    # 状态 4: 创建新沙箱
    return await _create_sandbox_for_user(user_id)


async def ping_user_sandbox(user_id: str) -> bool:
    """公开的沙箱健康检查，供中间件调用。"""
    proxy = SANDBOX_BACKENDS.get(user_id)
    if proxy is None:
        return False
    try:
        await asyncio.to_thread(proxy.execute, "echo ok")
        return True
    except Exception:
        return False


async def recreate_user_sandbox(user_id: str) -> SandboxBackendProxy:
    """
    公开的沙箱重建方法，供 SandboxHealthMiddleware 在检测到沙箱不可用时调用。

    创建新沙箱（含 skills 播种 + venv），替换 proxy 内的 backend，
    删除旧沙箱，更新 MongoDB 绑定。不包含 AGENTS.md 上传，由调用方负责。
    """
    proxy = SANDBOX_BACKENDS.get(user_id)
    return await _recreate_sandbox(user_id, proxy)


async def cleanup_user(user_id: str) -> None:
    """销毁某用户的沙箱。"""
    proxy = SANDBOX_BACKENDS.pop(user_id, None)
    if proxy is not None:
        await _try_delete_sandbox(proxy.id)

    _sandbox_collection().delete_one({"user_id": user_id})
    logger.info("用户 %s 沙箱已清理", user_id)


async def shutdown() -> None:
    """清理所有沙箱（含预热），释放资源。"""
    global _warm_reserve, _mongo_client
    logger.info("正在清理所有沙箱...")

    if _warm_reserve is not None:
        await _try_delete_sandbox(_warm_reserve.id)
        _warm_reserve = None

    for user_id in list(SANDBOX_BACKENDS.keys()):
        await cleanup_user(user_id)

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None

    logger.info("沙箱管理器已关闭")


# ---- 内部函数 ----


async def _reconnect_sandbox(sandbox_id: str) -> SandboxBackendProxy | None:
    """重连指定沙箱并验活，失败返回 None。"""
    try:
        from agent.backends.sandbox_setup import setup_sandbox
        sandbox_backend = await asyncio.to_thread(
            setup_sandbox, SANDBOX_CONFIG, sandbox_id=sandbox_id,
        )
        await asyncio.to_thread(sandbox_backend.execute, "echo ok")
    except Exception:
        return None
    return SandboxBackendProxy(sandbox_backend)


async def _claim_warm_reserve() -> SandboxBackendProxy | None:
    """
    取出预热沙箱并验活。

    预热沙箱可能闲置超时被平台销毁，认领前必须 ping。
    死沙箱丢弃并尝试删除。只要消耗了库存就触发后台补充。
    """
    global _warm_reserve
    async with _warm_lock:
        proxy = _warm_reserve
        _warm_reserve = None

    if proxy is None:
        return None

    asyncio.create_task(_replenish_warm())

    try:
        await asyncio.to_thread(proxy.execute, "echo ok")
        return proxy
    except Exception:
        logger.warning("预热沙箱 %s 已不可达，丢弃", proxy.id)
        await _try_delete_sandbox(proxy.id)
        return None


async def _try_delete_sandbox(sandbox_id: str) -> None:
    """尽力删除沙箱，失败仅记日志。"""
    try:
        from opensandbox import SandboxSync
        await asyncio.to_thread(SandboxSync.delete, sandbox_id)
    except Exception:
        logger.warning("删除沙箱 %s 失败", sandbox_id, exc_info=True)


async def _create_sandbox_for_user(user_id: str) -> SandboxBackendProxy:
    """创建新沙箱并绑定到用户。"""
    from agent.backends.sandbox_setup import setup_sandbox

    logger.info("为用户 %s 创建新沙箱...", user_id)
    sandbox_backend = await asyncio.to_thread(setup_sandbox, SANDBOX_CONFIG)
    sandbox_id = sandbox_backend.id

    proxy = SandboxBackendProxy(sandbox_backend)
    SANDBOX_BACKENDS[user_id] = proxy

    _sandbox_collection().update_one(
        {"user_id": user_id},
        {"$set": {
            "sandbox_id": sandbox_id,
            "updated_at": datetime.now(timezone.utc),
            "created_at": datetime.now(timezone.utc),
        }},
        upsert=True,
    )

    logger.info("用户 %s 沙箱创建完成: %s", user_id, sandbox_id)
    return proxy


async def _recreate_sandbox(
    user_id: str, existing_proxy: SandboxBackendProxy | None,
) -> SandboxBackendProxy:
    """重建沙箱（原沙箱不可达时）。"""
    from agent.backends.sandbox_setup import setup_sandbox

    logger.info("为用户 %s 重建沙箱...", user_id)
    sandbox_backend = await asyncio.to_thread(setup_sandbox, SANDBOX_CONFIG)
    sandbox_id = sandbox_backend.id

    if existing_proxy is not None:
        existing_proxy.replace_backend(sandbox_backend)
    else:
        existing_proxy = SandboxBackendProxy(sandbox_backend)
        SANDBOX_BACKENDS[user_id] = existing_proxy

    # 尝试删除旧沙箱
    old_doc = _sandbox_collection().find_one({"user_id": user_id})
    if old_doc and old_doc.get("sandbox_id"):
        await _try_delete_sandbox(old_doc["sandbox_id"])

    _sandbox_collection().update_one(
        {"user_id": user_id},
        {"$set": {
            "sandbox_id": sandbox_id,
            "updated_at": datetime.now(timezone.utc),
        }},
    )

    logger.info("用户 %s 沙箱重建完成: %s", user_id, sandbox_id)
    return existing_proxy


async def _replenish_warm() -> None:
    """后台补充预热沙箱（被认领后异步触发）。"""
    global _warm_reserve
    if _warm_reserve is not None:
        return

    try:
        from agent.backends.sandbox_setup import setup_sandbox
        sandbox_backend = await asyncio.to_thread(setup_sandbox, SANDBOX_CONFIG)
        async with _warm_lock:
            if _warm_reserve is None:
                _warm_reserve = SandboxBackendProxy(sandbox_backend)
                logger.info("后台补充预热沙箱就绪: %s", sandbox_backend.id)
    except Exception:
        logger.warning("后台补充预热沙箱失败", exc_info=True)
