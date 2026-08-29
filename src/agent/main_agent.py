"""
主 Agent 入口模块。

使用 DeepAgents `create_deep_agent` 将所有组件串联为一个可运行的
ERP 采购智能助手。采用 Graph Factory 模式：启动时预计算可复用组件，
每次请求基于 per-user 沙箱轻量创建 agent graph，实现用户级沙箱隔离。

使用方式:
    from agent.main_agent import precompute_agent_context, create_main_agent

    # 启动时
    precomputed = await precompute_agent_context()

    # 每次请求
    agent_graph = await create_main_agent(
        config,
        sandbox_backend=user_sandbox,
        precomputed=precomputed,
    )
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StoreBackend
from deepagents.backends.protocol import SandboxBackendProtocol

from agent.read_only_backend import ReadOnlyStoreBackend
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.runnables import RunnableConfig

from agent.config import (
    AGENTS_MD_FILENAME,
    CHECKPOINTER,
    DOWNLOAD_DIR,
    LOCAL_AGENTS_MD,
    MAIN_MODEL,
    SYSTEM_SKILLS_STORE_NAMESPACE,
    USER_SKILLS_STORE_BASE_NAMESPACE,
    STORE,
    SUMMARY_MODEL,
)
from agent.memory.prompts import system_prompt
from agent.middleware_config import (
    create_analyst_middleware,
    create_order_middleware,
)
from agent.middlewares.sandbox_breaker import SandboxCircuitBreakerMiddleware
from agent.middlewares.sandbox_health import SandboxHealthMiddleware
from agent.middlewares.skills_sync import SkillsSyncMiddleware
from agent.middlewares.memory_recall import MemoryRecallMiddleware
from agent.middlewares.tool_error import ToolErrorMiddleware
from agent.middlewares.tools_summarization import build_summarization_middleware
from agent.memory.database import memory_database
from agent.memory.policies import MemoryPolicy
from agent.memory.repository import MySQLMemoryRepository
from agent.memory.service import MemoryService
from agent.memory.tools import create_memory_tools
from agent.middlewares.user_skills_restore import UserSkillsRestoreMiddleware
from agent.schema import ProcurementContext
from agent.subagents.loader import load_subagent_configs, resolve_subagent_tools
from agent.tools.chart_generator import create_generate_chart_tool
from agent.tools.hitl_tools import request_order_info
from agent.tools.assign_skill import create_assign_skill_tool
from agent.tools.download_sandbox_file import create_download_tool
from agent.tools.mcp_client import load_mcp_tools
from agent.tools.web_search import web_search


def _setup_logging() -> None:
    env = os.environ.get("APP_ENV", "development")
    if env == "production":
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            filename="erp_agent.log",
            filemode="a",
        )
    else:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stdout,
        )

_setup_logging()
logger = logging.getLogger(__name__)


# ============================================================
# PrecomputedContext — 启动时预计算的可复用组件
# ============================================================

@dataclass
class PrecomputedContext:
    """Phase 2/3/5 预计算结果的不可变容器。

    仅包含不依赖 sandbox_backend 的组件（MCP 工具、图表工具、YAML 配置）。
    sandbox 依赖的工具（assign_skill、download_sandbox_file）和 backend 依赖的
    中间件在 create_main_agent() 中按请求动态创建。
    """
    all_mcp_tools: list = field(default_factory=list)
    analyst_mcp_tools: list = field(default_factory=list)
    order_mcp_tools: list = field(default_factory=list)
    chart_mcp_tools: list = field(default_factory=list)
    extra_mcp_tools: list = field(default_factory=list)
    generate_visualization: object = None
    raw_subagent_configs: list = field(default_factory=list)


async def precompute_agent_context() -> PrecomputedContext:
    """Phase 2/3/5 预计算，启动时执行一次，所有请求复用。"""
    logger.info("=== 预计算 Agent 上下文（Phase 2/3/5）===")

    # Phase 2: MCP 工具加载
    logger.info("Phase 2: 加载 MCP 工具...")
    try:
        all_mcp_tools, analyst_mcp_tools, order_mcp_tools, chart_mcp_tools = (
            await load_mcp_tools()
        )
    except Exception:
        logger.exception("MCP 工具加载失败")
        raise RuntimeError("MCP 工具加载失败，无法预计算")

    # Phase 3: 可视化工具合并
    logger.info("Phase 3: 合并可视化工具 (26→1)...")
    generate_visualization, extra_mcp_tools = create_generate_chart_tool(chart_mcp_tools)
    if extra_mcp_tools:
        logger.info(f"  保留独立工具: {[t.name for t in extra_mcp_tools]}")

    # Phase 5: 子 Agent YAML 配置加载
    logger.info("Phase 5: 加载子 Agent YAML 配置...")
    raw_configs = load_subagent_configs()
    if not raw_configs:
        logger.warning("  未找到任何子 Agent 配置")
    else:
        logger.info(f"  已加载 {len(raw_configs)} 个子 Agent 配置")

    logger.info("=== 预计算完成 ===")
    return PrecomputedContext(
        all_mcp_tools=all_mcp_tools,
        analyst_mcp_tools=analyst_mcp_tools,
        order_mcp_tools=order_mcp_tools,
        chart_mcp_tools=chart_mcp_tools,
        extra_mcp_tools=extra_mcp_tools,
        generate_visualization=generate_visualization,
        raw_subagent_configs=raw_configs,
    )


# ============================================================
# create_main_agent — 每次请求基于 per-user 沙箱创建 agent graph
# ============================================================

async def create_main_agent(
    config: RunnableConfig,
    *,
    sandbox_backend: SandboxBackendProtocol,
    precomputed: PrecomputedContext,
):
    """创建 ERP 采购智能助手的 per-request graph factory。

    每次请求调用，使用预计算的 MCP 工具/YAML 配置 + 外部传入的 per-user 沙箱，
    轻量创建 agent graph。SandboxBackendProxy 保证沙箱热替换不丢引用。

    Args:
        config: LangGraph RunnableConfig，含 thread_id + user_id。
        sandbox_backend: per-user 沙箱后端（SandboxBackendProxy）。
        precomputed: 启动时预计算的 MCP 工具/YAML 配置。
    """
    user_id = config["configurable"]["user_id"]
    logger.info(f"=== 为用户 {user_id} 创建 Agent Graph ===")

    # ---- Phase 1: CompositeBackend factory ----
    # 每次请求重建 CompositeBackend（StoreBackend 依赖 runtime），
    # 内部 sandbox_backend 是 SandboxBackendProxy（热替换不丢引用）。
    def backend_factory(runtime):
        return CompositeBackend(
            default=sandbox_backend,
            routes={
                # 系统技能路由（共享，所有用户只读；仅 SkillsSyncMiddleware 经 store.put 写入）
                "/persisted-skills/system/": ReadOnlyStoreBackend(
                    runtime=runtime,
                    namespace=lambda rt: SYSTEM_SKILLS_STORE_NAMESPACE,
                ),
                # 用户技能路由（按 user_id 隔离）
                "/persisted-skills/": StoreBackend(
                    runtime=runtime,
                    namespace=lambda rt: USER_SKILLS_STORE_BASE_NAMESPACE
                        + (getattr(rt.runtime.context, 'user_id', 'Quan'),),
                ),
            },
        )

    # ---- Phase 1.4: 上传 AGENTS.md 到沙箱 ----
    logger.info("Phase 1.4: 上传 AGENTS.md 到沙箱...")
    ag_md_content = LOCAL_AGENTS_MD.read_text(encoding="utf-8")
    sandbox_backend.upload_files([("/AGENTS.md", ag_md_content.encode("utf-8"))])

    # ---- Phase 2/3/5: 使用预计算结果 ----
    logger.info("Phase 2-5: 使用预计算的 MCP 工具 + 图表工具 + YAML 配置...")
    generate_visualization = precomputed.generate_visualization
    extra_mcp_tools = precomputed.extra_mcp_tools

    # ---- Phase 3.6: 创建 sandbox 依赖工具（per-request）----
    logger.info("Phase 3.6: 创建 sandbox 依赖工具...")
    assign_skill = create_assign_skill_tool(
        sandbox_backend,
        store=STORE,
        user_id=user_id,
    )
    download_sandbox_file = create_download_tool(sandbox_backend, DOWNLOAD_DIR)

    # ---- Phase 3.7: Memory v2 受控记忆工具（仅当 Memory v2 启用且 WRITE=1，review #8/I）----
    memory_tools = []
    if memory_database.can_write:
        memory_service = MemoryService(
            MySQLMemoryRepository(memory_database.session_factory),
            MemoryPolicy(),
        )
        memory_tools = create_memory_tools(
            memory_service, user_id,
            thread_id=str(config["configurable"].get("thread_id", "")),
        )
        logger.info(f"  Memory v2 写入已启用，注入 {len(memory_tools)} 个记忆工具")

    # ---- Phase 4: 构建工具池 ----
    logger.info("Phase 4: 构建工具池...")
    available_tools = (
        list(precomputed.analyst_mcp_tools)
        + list(precomputed.order_mcp_tools)
        + list(extra_mcp_tools)
        + [generate_visualization]
        + [web_search]
        + [request_order_info]
        + [assign_skill]
        + [download_sandbox_file]
        + memory_tools
    )
    logger.info(f"  工具池: {len(available_tools)} 个工具")

    # ---- Phase 6: 子 Agent 中间件（analyst 依赖 backend_factory）----
    logger.info("Phase 6: 创建子 Agent 中间件...")
    extra_middleware = {
        "procurement-analyst": create_analyst_middleware(SUMMARY_MODEL, backend_factory),
        "procurement-order": create_order_middleware(),
    }

    # ---- Phase 7: 子 Agent 工具解析 ----
    logger.info("Phase 7: 解析子 Agent 工具名称...")
    subagents = resolve_subagent_tools(
        precomputed.raw_subagent_configs,
        available_tools,
        extra_middleware=extra_middleware,
    )
    logger.info(f"  已解析 {len(subagents)} 个子 Agent")

    # ---- Phase 8: 主 Agent 中间件栈 ----
    logger.info("Phase 8: 构建主 Agent 中间件栈...")
    # Memory v2（方案 5.7）：旧 Markdown 记忆路径已删除（无旧记忆中间件注册，
    # 身份信息只进 runtime.context，不写 message state）。
    # v2 关闭时没有长期记忆（禁止回退 Markdown，21.1）
    memory_v2_active = memory_database.initialized
    main_middleware = [
        # 1. 沙箱健康守护：每次 agent step 前 ping → 失败自动恢复
        SandboxHealthMiddleware(
            sandbox_backend=sandbox_backend,
            user_id=user_id,
            agents_md_content=ag_md_content.encode("utf-8"),
        ),
        # 2. 工具错误捕获：wrap_tool_call → ToolMessage(status="error")，防止单工具崩溃
        ToolErrorMiddleware(),
        # 3. 技能同步（本地 → 沙箱）
        SkillsSyncMiddleware(sandbox_backend, store=STORE),
        # 4. 持久化技能恢复（StoreBackend → 沙箱）
        UserSkillsRestoreMiddleware(sandbox_backend, system_namespace=SYSTEM_SKILLS_STORE_NAMESPACE, user_id=user_id),
        # 5. 对话摘要
        build_summarization_middleware(backend_factory, SUMMARY_MODEL),
        # 7. 沙箱熔断：连续沙箱错误 ≥ 阈值 → jump_to=end
        SandboxCircuitBreakerMiddleware(),
        # 8. 调用限制
        ModelCallLimitMiddleware(run_limit=50),
        ToolCallLimitMiddleware(run_limit=200),
    ]
    if memory_v2_active and memory_database.can_read:
        # 6. Memory v2 记忆召回（v2 active + READ=1 时；model-call-time 临时注入，
        # 不写 state/checkpoint，方案 18.6）
        memory_recall_middleware = MemoryRecallMiddleware(
            MySQLMemoryRepository(memory_database.session_factory)
        )
        main_middleware.insert(5, memory_recall_middleware)

    # ---- Phase 9: create_deep_agent ----
    logger.info("Phase 9: 创建 Deep Agent...")
    # 主 Agent 工具池（review F：memory_tools 必须真正传入 create_deep_agent）
    main_tools = [web_search, assign_skill, download_sandbox_file] + memory_tools
    agent_graph = create_deep_agent(
        model=MAIN_MODEL,
        system_prompt=system_prompt,
        skills=["/skills/main/"],
        memory=[AGENTS_MD_FILENAME],
        tools=main_tools,
        subagents=subagents,
        middleware=main_middleware,
        backend=backend_factory,
        store=STORE,
        checkpointer=CHECKPOINTER,
        context_schema=ProcurementContext,
    )

    logger.info(f"=== 用户 {user_id} Agent Graph 创建完成 ===")
    return agent_graph
