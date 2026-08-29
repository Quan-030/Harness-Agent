"""
对话 API 模块

提供流式对话接口、中断恢复接口和会话状态查询接口
"""

import json
import logging
import uuid
import re
import os
import tempfile
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator

logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent.schema import (
    ChatRequest,
    ChatResponse,
    Message,
)
from api_view.agent_loader import agent_loader
from agent.config import MEMORY_BACKGROUND_JOBS_ENABLED
from agent.memory.database import memory_database


# 创建路由
router = APIRouter()

# 调试日志文件路径（在项目根目录的 temp 文件夹下）
DEBUG_LOG_DIR = os.path.join(tempfile.gettempdir(), "deepagent_debug")
os.makedirs(DEBUG_LOG_DIR, exist_ok=True)


def get_debug_log_path(thread_id: str) -> str:
    """获取当前会话的调试日志文件路径"""
    safe_id = thread_id.replace("/", "_").replace("\\", "_")[:50]
    return os.path.join(DEBUG_LOG_DIR, f"stream_{safe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")


def write_debug_log(filepath: str, event_type: str, data: dict, raw_token: object = None):
    """写入调试日志"""
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] {event_type}\n")
            f.write(f"  data: {json.dumps(data, ensure_ascii=False, default=str)[:2000]}\n")
            if raw_token is not None:
                token_info = {}
                if hasattr(raw_token, 'type'):
                    token_info['type'] = raw_token.type
                if hasattr(raw_token, 'name'):
                    token_info['name'] = raw_token.name
                if hasattr(raw_token, 'content'):
                    c = raw_token.content
                    if isinstance(c, str):
                        token_info['content_preview'] = c[:500]
                    elif isinstance(c, list):
                        token_info['content_len'] = len(c)
                        token_info['content_preview'] = str(c)[:500]
                    else:
                        token_info['content_preview'] = str(c)[:500]
                if hasattr(raw_token, 'tool_call_chunks') and raw_token.tool_call_chunks:
                    token_info['tool_call_chunks'] = str(raw_token.tool_call_chunks)[:500]
                if hasattr(raw_token, 'id'):
                    token_info['id'] = raw_token.id
                f.write(f"  token: {json.dumps(token_info, ensure_ascii=False, default=str)[:2000]}\n")
            f.write("\n")
    except Exception:
        pass  # 调试日志失败不影响主流程


def extract_subagent_name(namespace: tuple) -> str:
    """
    从 namespace 中提取子代理名称
    """
    for segment in namespace:
        if segment.startswith("tools:"):
            return segment.replace("tools:", "")
    return "main"


def is_likely_uuid(text: str) -> bool:
    """
    判断文本是否大概率是 UUID（而非有意义的回复内容）
    例如：3576bba4-42e5-a769-c2f7-ea8444829951
    """
    # UUID 格式：8-4-4-4-12
    uuid_pattern = r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    return bool(re.match(uuid_pattern, text.strip()))


def extract_content_from_token(token) -> str:
    """
    从 token 中提取文本内容

    处理 token.content 可能是字符串、列表或其他类型的情况
    过滤掉明显是 UUID 或无关标识符的内容
    """
    if not hasattr(token, 'content'):
        return ""

    content = token.content

    # 如果 content 是字符串，直接返回
    if isinstance(content, str):
        # 过滤纯 UUID 内容
        if is_likely_uuid(content):
            return ""
        return content

    # 如果 content 是列表，尝试提取其中的文本
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                # 结构化内容：只提取 text 类型，忽略 image_url 等
                item_type = item.get("type", "")
                if item_type == "text":
                    t = item.get("text", "")
                    if not is_likely_uuid(t):
                        text_parts.append(t)
                elif item_type == "image_url":
                    # 图片 URL 不提取为文本，由 serialize_tool_result 处理
                    pass
                elif "text" in item:
                    t = item["text"]
                    if not is_likely_uuid(str(t)):
                        text_parts.append(str(t))
                elif "content" in item:
                    text_parts.append(str(item["content"]))
            else:
                text_parts.append(str(item))
        return ''.join(text_parts)

    # 其他类型转换为字符串
    return str(content) if content is not None else ""


def _tool_result_has_error(content) -> bool | None:
    """结构化检查工具返回是否携带错误（review 第三轮，非文本猜测）。

    依据 ERP 工具返回契约（src/mcp_server/tools/order_tools.py）：
    - 业务失败：返回 {"error": ...} dict（MCP 层序列化为 JSON 文本）
    - 成功：返回 data dict（无 error 键）
    无法解析为 dict 时返回 None（无法判定 → 不写 succeeded 字段）。

    Args:
        content: ToolMessage.content，可能是 str（JSON 文本）、dict 或 content blocks 列表。
    Returns:
        True=含 error 键；False=可判定且无 error 键；None=无法结构化判定。
    """
    data = None
    if isinstance(content, str):
        text = content.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
            except (ValueError, TypeError):
                return None
    elif isinstance(content, dict):
        data = content
    elif isinstance(content, list):
        # content blocks（langchain_mcp_adapters：{"type": "text", "text": ...}）
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", "")).strip()
                if text.startswith("{") and text.endswith("}"):
                    try:
                        data = json.loads(text)
                        break
                    except (ValueError, TypeError):
                        pass
    if not isinstance(data, dict):
        return None
    return "error" in data


def serialize_tool_result(content) -> dict:
    """
    将工具执行结果序列化为前端可用的结构化数据

    处理 tool 消息的 content，可能是字符串、列表（包含文本和图片）
    提取文本和图片数据，返回结构化字典
    """
    result = {"text": "", "images": []}

    if isinstance(content, str):
        result["text"] = content
        # 提取 markdown 图片语法
        for match in re.finditer(r'!\[.*?\]\((.*?)\)', content):
            url = match.group(1)
            if url not in result["images"]:
                result["images"].append(url)

    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "text":
                    t = item.get("text", "")
                    text_parts.append(t)
                elif item_type == "image_url":
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                    if url:
                        result["images"].append(url)
                elif item_type == "image":
                    image_data = item.get("data", "") or item.get("image_data", "")
                    if image_data:
                        result["images"].append(image_data)
                elif "text" in item:
                    text_parts.append(item["text"])
                elif "content" in item:
                    text_parts.append(str(item["content"]))
                else:
                    text_parts.append(str(item))
            else:
                text_parts.append(str(item))
        result["text"] = ''.join(text_parts)
    else:
        result["text"] = str(content) if content is not None else ""

    # 从文本中提取所有图片 URL（包括 markdown 语法和直链）
    # 支持常见图片托管域名（即使 URL 不以图片扩展名结尾）
    if result["text"]:
        for match in re.finditer(r'!\[.*?\]\((.*?)\)', result["text"]):
            url = match.group(1)
            if url not in result["images"]:
                result["images"].append(url)
        # 也检测 data:image base64
        for match in re.finditer(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', result["text"]):
            url = match.group(0)
            if url not in result["images"]:
                result["images"].append(url)

    return result


def create_sse_message(data: dict) -> str:
    """创建 SSE 格式的消息"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============================================================
# 中断恢复请求模型
# ============================================================

class ResumeRequest(BaseModel):
    """中断恢复请求体，resume 字段的格式取决于中断类型：
    - 数据补充中断: {"supplement": "用户自由文本输入"}
    - HITL 审批中断: {"decisions": [{"type": "approve"}]} 或 [{"type": "reject"}]
    """
    resume: dict
    user_id: str = "Quan"
    resume_id: str = Field(min_length=8, max_length=64)


# HITL 中断签发的 resume_id（进程内短期存储，方案 18.6：可信 resume_id，不接受伪造）
_RESUME_TOKEN_TTL = timedelta(minutes=5)
_resume_tokens: dict[str, dict] = {}


# ============================================================
# 流式对话核心逻辑
# ============================================================

async def stream_chat_response(
    message: str = None,
    thread_id: str = None,
    resume_data: dict = None,
    user_id: str = "Quan",
    invocation_id: str = "",
    resume_id: str | None = None,
) -> AsyncIterator[str]:
    """
    流式生成对话响应，支持 Human-in-the-Loop 中断与恢复。

    两种调用模式：
    1. 初始对话 — 传入 message（用户消息）
    2. 中断恢复 — 传入 resume_data（Command.resume 的值）

    当 Agent 触发中断时（request_order_info 数据补充 / order_create|update HITL 审批），
    流会发送 interrupt 事件后结束。前端收集用户决策后通过 /resume 端点恢复。

    同时累积完整的展示消息列表（包含子代理消息），在流结束后存入 MongoDB。
    """
    # Memory v2（方案 18.6）：invocation_id 服务端生成，resume_id 为 HITL 可信恢复标识；
    # 只进 runtime.context，绝不写入 state/checkpoint
    context = {
        "user_id": user_id,
        "username": user_id,
        "thread_id": thread_id,
        "invocation_id": invocation_id,
        "resume_id": resume_id,
    }
    config = agent_loader.create_config(thread_id, user_id=user_id)

    # 获取该用户的 per-user agent graph（沙箱缓存 + 预计算组件）
    agent_graph = await agent_loader.get_agent_for_user(user_id)
    collected_content = ""
    # 工具调用栈，支持嵌套工具调用（主代理调 task → 子代理调 generate_chart）
    tool_call_stack = []

    # ---- 根据模式构建 input 和 display_messages ----
    if resume_data is not None:
        # 恢复模式：加载已有展示消息，用 Command(resume=...) 恢复
        existing = await agent_loader.get_display_messages(thread_id) or []
        display_messages = existing
        current_input = Command(resume=resume_data)
    else:
        # 初始模式：新建展示消息，用普通消息作为 input
        current_input = {"messages": [{"role": "user", "content": message}]}
        display_messages = [
            {
                "id": f"user-{uuid.uuid4()}",
                "role": "user",
                "content": message
            }
        ]

    # 创建调试日志文件
    debug_log = get_debug_log_path(thread_id)

    def _last_display_is_assistant():
        return (display_messages and
                display_messages[-1]["role"] == "assistant")

    try:
        write_debug_log(debug_log, "STREAM_START", {
            "message": message,
            "thread_id": thread_id,
            "is_resume": resume_data is not None,
        })

        # 流式调用 agent.astream()
        # stream_mode=["messages", "values"] — messages 流显示 + values 流检测中断
        async for chunk in agent_graph.astream(
            input=current_input,
            config=config,
            context=context,
            stream_mode=["messages", "values"],
            subgraphs=True, # 启用子代理流式输出
            version="v2",
        ):
            chunk_type = chunk.get("type")

            # ---- 值流（中断检测，必须在 messages 处理之前）----
            if chunk_type == "values" and chunk.get("interrupts"):
                interrupts = chunk["interrupts"]
                write_debug_log(debug_log, "INTERRUPT_DETECTED", {
                    "count": len(interrupts),
                })

                # Memory v2：每次中断签发可信 resume_id（方案 18.6），
                # 随 interrupt 事件下发，resume 请求必须原样带回校验
                resume_token = uuid.uuid4().hex
                _resume_tokens[thread_id] = {
                    "resume_id": resume_token,
                    "expires_at": datetime.now(timezone.utc) + _RESUME_TOKEN_TTL,
                }

                for interrupt in interrupts:
                    interrupt_value = interrupt.value

                    if "action_requests" in interrupt_value:
                        # ---- 第 2 层：HITL 审批中断（interrupt_on 配置）----
                        yield create_sse_message({
                            "type": "interrupt",
                            "interrupt_type": "hitl_approval",
                            "action_requests": interrupt_value["action_requests"],
                            "review_configs": interrupt_value.get("review_configs", []),
                            "thread_id": thread_id,
                            "resume_id": resume_token,
                        })
                        write_debug_log(debug_log, "INTERRUPT_HITL", {
                            "actions": [a["name"] for a in interrupt_value["action_requests"]],
                        })

                    elif interrupt_value.get("type") == "order_info_request":
                        # ---- 第 1 层：数据补充中断（request_order_info 工具）----
                        yield create_sse_message({
                            "type": "interrupt",
                            "interrupt_type": "order_info_supplement",
                            "missing_fields": interrupt_value["missing_fields"],
                            "collected_data": interrupt_value["collected_data"],
                            "thread_id": thread_id,
                            "resume_id": resume_token,
                        })
                        write_debug_log(debug_log, "INTERRUPT_SUPPLEMENT", {
                            "missing_fields": interrupt_value["missing_fields"],
                        })

                    else:
                        # 未知中断类型，透传原始值
                        yield create_sse_message({
                            "type": "interrupt",
                            "interrupt_type": "unknown",
                            "interrupt_value": str(interrupt_value)[:2000],
                            "thread_id": thread_id,
                            "resume_id": resume_token,
                        })

                # 兜底：将所有 calling 状态的工具标记为 done
                for dm in display_messages:
                    if dm["role"] == "tool" and dm["tool_status"] == "calling":
                        dm["tool_status"] = "done"

                # 清理空的 assistant 消息
                cleaned = [
                    dm for dm in display_messages
                    if not (dm["role"] == "assistant" and not dm.get("content"))
                ]

                # 保存部分展示消息到 MongoDB（中断前的消息状态）
                # 注意：resume 模式下 display_messages 已含历史，保存时会覆盖旧记录
                await agent_loader.save_display_messages(thread_id, cleaned)
                write_debug_log(debug_log, "SAVE_DISPLAY_INTERRUPT", {
                    "thread_id": thread_id,
                    "message_count": len(cleaned),
                })

                # 发送 done 事件标记流结束（前端由此知道可以展示中断 UI）
                yield create_sse_message({
                    "type": "done",
                    "thread_id": thread_id,
                    "content": collected_content,
                    "interrupted": True,
                })
                return  # ← 结束本次流，等待前端 POST /resume

            # ---- 消息流（token / tool_call / tool_result）----
            if chunk_type != "messages":
                continue

            token, metadata = chunk["data"]
            namespace = chunk.get("ns", ())

            # 判断消息来源
            is_subagent = any(s.startswith("tools:") for s in namespace)
            source = extract_subagent_name(namespace) if is_subagent else "main"

            write_debug_log(debug_log, "RAW_CHUNK", {
                "ns": list(namespace),
                "source": source,
            }, raw_token=token)

            # 处理工具调用
            if hasattr(token, 'tool_call_chunks') and token.tool_call_chunks:
                for tool_chunk in token.tool_call_chunks:
                    # 工具开始调用
                    if tool_chunk.get('name'):
                        tool_id = str(uuid.uuid4())
                        new_tool = {
                            "id": tool_id,
                            "name": tool_chunk['name'],
                            "args": "",
                            "source": source
                        }
                        tool_call_stack.append(new_tool)

                        yield create_sse_message({
                            "type": "tool_start",
                            "tool_call_id": tool_id,
                            "tool_name": tool_chunk['name'],
                            "source": source
                        })

                        # 添加到展示消息
                        display_messages.append({
                            "id": tool_id,
                            "role": "tool",
                            "tool_name": tool_chunk['name'],
                            "args": "",
                            "text": "",
                            "images": [],
                            "source": source,
                            "tool_status": "calling"
                        })

                        write_debug_log(debug_log, "TOOL_START", {
                            "name": tool_chunk['name'],
                            "source": source,
                            "stack_depth": len(tool_call_stack)
                        })

                    # 工具参数
                    if tool_chunk.get('args'):
                        args_str = tool_chunk['args']
                        if tool_call_stack:
                            tool_call_stack[-1]["args"] += args_str

                        yield create_sse_message({
                            "type": "tool_args",
                            "args": args_str,
                            "source": source
                        })

                        # 更新展示消息中对应工具的 args（从后向前找最后一个 calling 的 tool）
                        for dm in reversed(display_messages):
                            if dm["role"] == "tool" and dm["tool_status"] == "calling":
                                dm["args"] += args_str
                                break

            # 处理工具执行结果
            if hasattr(token, 'type') and token.type == "tool":
                tool_name = getattr(token, 'name', '未知工具')
                result_content = getattr(token, 'content', '')

                # 序列化工具结果为结构化数据
                serialized = serialize_tool_result(result_content)

                # 从栈中弹出对应工具（栈顶即为当前完成的工具）
                finished_tool = tool_call_stack.pop() if tool_call_stack else None
                tool_id = finished_tool["id"] if finished_tool else ""

                yield create_sse_message({
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "text": serialized["text"],
                    "images": serialized["images"],
                    "source": source
                })

                # 更新展示消息中对应工具的结果
                # Memory v2（review 第三轮）：producer 端生成可信 succeeded——
                # 来源（均为结构化状态，非文本猜测）：
                # 1. ToolMessage.status="error"（ToolErrorMiddleware 捕获的异常，
                #    含 langchain_mcp_adapters 对 MCP isError=true 抛出的 ToolException）
                #    → succeeded=False
                # 2. 否则按 ERP 工具返回契约结构化检查（order_tools.py：业务失败返回
                #    {"error": ...} dict，成功返回 data dict）→ 可判定时写入布尔值
                # 3. 无法结构化判定（纯文本等）→ 不写字段（无法确认，不生成候选）
                tool_status = getattr(token, "status", "") or ""
                succeeded = None
                if tool_status == "error":
                    succeeded = False
                else:
                    error_detected = _tool_result_has_error(result_content)
                    if error_detected is not None:
                        succeeded = not error_detected
                for dm in reversed(display_messages):
                    if dm["role"] == "tool" and dm["id"] == tool_id:
                        dm["text"] = serialized["text"]
                        dm["images"] = serialized["images"]
                        dm["tool_status"] = "done"
                        if succeeded is not None:
                            dm["succeeded"] = succeeded
                        break

                write_debug_log(debug_log, "TOOL_RESULT", {
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "text_preview": serialized["text"][:300],
                    "images_count": len(serialized["images"]),
                    "source": source,
                    "stack_remaining": len(tool_call_stack)
                })

                # 发送 tool_end 事件，标记工具调用结束
                yield create_sse_message({
                    "type": "tool_end",
                    "tool_name": tool_name,
                    "tool_call_id": tool_id,
                    "source": source
                })

                # 确保展示消息中的工具标记为 done（兜底）
                for dm in reversed(display_messages):
                    if dm["role"] == "tool" and dm["id"] == tool_id and dm["tool_status"] == "calling":
                        dm["tool_status"] = "done"
                        break

            # 处理 AI 文本内容
            # 注意：ToolMessage（type == "tool"）的内容是工具执行结果，已在上面
            # tool_result 事件中处理。这里必须跳过，否则工具结果会被当作 AI 文本重复输出。
            content_text = extract_content_from_token(token)
            has_tool_calls = hasattr(token, 'tool_call_chunks') and token.tool_call_chunks
            is_tool_result = hasattr(token, 'type') and token.type == "tool"

            if content_text and not has_tool_calls and not is_tool_result:
                collected_content += content_text
                yield create_sse_message({
                    "type": "token",
                    "content": content_text,
                    "source": source
                })
                write_debug_log(debug_log, "TOKEN", {
                    "content": content_text[:200],
                    "source": source
                })

                # 更新展示消息：如果最后一条是 assistant 则追加，否则新建
                if _last_display_is_assistant():
                    display_messages[-1]["content"] += content_text
                    display_messages[-1]["source"] = source
                else:
                    display_messages.append({
                        "id": f"assistant-{uuid.uuid4()}",
                        "role": "assistant",
                        "content": content_text,
                        "source": source
                    })

        # ---- 流正常结束（无中断）----
        # 兜底：将所有 calling 状态的工具标记为 done
        for dm in display_messages:
            if dm["role"] == "tool" and dm["tool_status"] == "calling":
                dm["tool_status"] = "done"

        # 清理空的 assistant 消息
        display_messages = [
            dm for dm in display_messages
            if not (dm["role"] == "assistant" and not dm.get("content"))
        ]

        # 保存完整展示消息到 MongoDB（包含子代理消息）
        # 多轮对话：追加到已有消息，而非覆盖（每轮调用 save 时 display_messages 仅含当前轮）
        if resume_data is None:
            # 初始对话：追加到已有历史
            existing = await agent_loader.get_display_messages(thread_id) or []
            all_messages = existing + display_messages
        else:
            # resume 模式：display_messages 已包含历史（load 时获取），直接保存
            all_messages = display_messages
        await agent_loader.save_display_messages(thread_id, all_messages)
        write_debug_log(debug_log, "SAVE_DISPLAY", {
            "thread_id": thread_id,
            "total_count": len(all_messages)
        })

        # Memory v2（方案 6.1）：唯一入队边界——确认三类 ID 后入队 extract_memory（幂等）
        if memory_database.can_write and MEMORY_BACKGROUND_JOBS_ENABLED:
            from api_view.conversation_completion import (
                ConversationCompletionService,
                extract_ids_from_display_messages,
            )
            from agent.memory.repository import MySQLMemoryRepository

            user_message_id, assistant_message_id = extract_ids_from_display_messages(
                all_messages
            )
            checkpoint_id = None
            try:
                state = await agent_graph.aget_state(config)
                checkpoint_id = (state.config or {}).get("configurable", {}).get(
                    "checkpoint_id"
                )
            except Exception:
                logger.exception("获取 checkpoint_id 失败，本轮回合由 reconciliation 补建")
            if user_message_id and assistant_message_id and checkpoint_id:
                completion = ConversationCompletionService(
                    MySQLMemoryRepository(memory_database.session_factory)
                )
                try:
                    await completion.enqueue_extract(
                        user_id=user_id,
                        thread_id=thread_id,
                        user_message_id=user_message_id,
                        assistant_message_id=assistant_message_id,
                        checkpoint_id=checkpoint_id,
                    )
                except Exception as exc:
                    # review #5（方案 19.2）：enqueue 失败不撤回已发送的回复，
                    # 不破坏 SSE done；只记异常类型（不打印 SQL/DSN/凭据），
                    # 遗漏 Job 由 reconciliation 补建
                    logger.error(
                        "Memory extract enqueue failed: type=%s thread=%s",
                        type(exc).__name__, thread_id,
                    )
            else:
                logger.warning(
                    "Memory v2: 本轮 ID 不完整（user=%s am=%s cp=%s），不创建不完整 Job",
                    bool(user_message_id), bool(assistant_message_id), bool(checkpoint_id),
                )

        # 流结束，发送 done 事件
        write_debug_log(debug_log, "STREAM_DONE", {
            "thread_id": thread_id,
            "total_content_len": len(collected_content)
        })

        yield create_sse_message({
            "type": "done",
            "thread_id": thread_id,
            "content": collected_content
        })

    except Exception as e:
        write_debug_log(debug_log, "STREAM_ERROR", {"error": str(e)})
        yield create_sse_message({
            "type": "error",
            "message": str(e)
        })


# ============================================================
# API 端点
# ============================================================

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话接口

    接收用户消息，返回 SSE 流式响应
    支持实时显示 AI 生成的内容和工具调用信息
    检测到 Human-in-the-Loop 中断时会发送 interrupt 事件
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    user_id = request.user_id
    # Memory v2（方案 18.6）：每次 API 调用由服务端生成新 invocation_id
    invocation_id = uuid.uuid4().hex

    return StreamingResponse(
        stream_chat_response(
            message=request.message, thread_id=thread_id, user_id=user_id,
            invocation_id=invocation_id,
        ),
        media_type="text/event-stream",  # text/json, text/html, image/jpg
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/chat/{thread_id}/resume")
async def chat_resume(thread_id: str, request: ResumeRequest):
    """
    中断恢复接口

    当中断发生后，前端收集用户决策/补充数据，通过此端点恢复 Agent 执行。
    request.resume 的格式取决于中断类型：
    - 数据补充: {"supplement": "用户输入的补充信息"}
    - HITL 审批: {"decisions": [{"type": "approve"}]} 或 [{"type": "reject"}]

    resume_id 由中断时服务端签发（随 interrupt 事件下发），此处校验可信性。
    """
    # Memory v2（方案 18.6）：校验可信 resume_id；不接受客户端任意伪造值
    entry = _resume_tokens.get(thread_id)
    if entry is None or entry["resume_id"] != request.resume_id:
        raise HTTPException(status_code=400, detail="resume_id 无效或已过期，请重新发起对话")
    if entry["expires_at"] < datetime.now(timezone.utc):
        _resume_tokens.pop(thread_id, None)
        raise HTTPException(status_code=400, detail="resume_id 已过期，请重新发起对话")
    _resume_tokens.pop(thread_id, None)  # 一次性使用

    # 每次 resume 调用生成新的 invocation_id（不复用普通输入 invocation 的 snapshot）
    invocation_id = uuid.uuid4().hex

    return StreamingResponse(
        stream_chat_response(
            thread_id=thread_id, resume_data=request.resume, user_id=request.user_id,
            invocation_id=invocation_id, resume_id=request.resume_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/chat/{thread_id}")
async def get_chat_state(thread_id: str):
    """获取会话状态"""
    try:
        messages = await agent_loader.get_current_messages(thread_id)

        message_list = []
        for i, msg in enumerate(messages):
            message = Message(
                id=f"msg-{i}",
                role=msg.get("role", "assistant"),
                content=msg.get("content", ""),
                created_at=datetime.now(),
                tool_calls=msg.get("tool_calls"),
                tool_call_id=msg.get("tool_call_id"),
            )
            message_list.append(message)

        return ChatResponse(
            thread_id=thread_id,
            messages=message_list
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/{thread_id}/history")
async def get_chat_history(
    thread_id: str,
    limit: int = Query(50, ge=1, le=100)
):
    """获取会话历史状态列表"""
    try:
        states = await agent_loader.get_state_history(thread_id, limit)

        return {
            "thread_id": thread_id,
            "states": [
                {
                    "config": state.config,
                    "values": state.values,
                    "created_at": state.created_at,
                    "parent_config": state.parent_config
                }
                for state in states
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
