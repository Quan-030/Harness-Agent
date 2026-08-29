<template>
  <div class="chat-area">
    <!-- 消息列表 -->
    <div class="message-list" ref="messageListRef">
      <!-- 空状态 -->
      <div v-if="displayMessages.length === 0" class="empty-state">
        <div class="empty-icon"><img src="../assets/quanmotor-logo.svg" alt="QuanMotor" /></div>
        <h2 class="empty-title">Harness Agent</h2>
        <p class="empty-tagline">基于 Harness Engineering 的智能助手</p>
        <div class="example-chips">
          <div class="example-chip">数据分析和图表生成</div>
          <div class="example-chip">网络搜索和信息查询</div>
          <div class="example-chip">代码编写和调试</div>
          <div class="example-chip">文档处理和写作</div>
        </div>
      </div>

      <!-- 消息列表（user / assistant / tool 按时间顺序混合展示）-->
      <div v-else class="messages">
        <MessageItem
          v-for="(message, index) in displayMessages"
          :key="message.id || index"
          :message="message"
          :is-streaming="isStreamingForMessage(message)"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import MessageItem from './MessageItem.vue'

/**
 * 对话区域组件
 *
 * 显示消息列表（user / assistant / tool 按时间顺序混合展示）
 * 当 showToolCalls 为 false 时，过滤掉 tool 消息只显示 AI 回复
 */

const props = defineProps({
  messages: {
    type: Array,
    default: () => []
  },
  streaming: {
    type: Boolean,
    default: false
  },
  showToolCalls: {
    type: Boolean,
    default: true
  }
})

// 根据开关过滤展示的消息
const displayMessages = computed(() => {
  if (props.showToolCalls) {
    return props.messages
  }
  // 过滤掉 tool 消息，只保留 user 和 assistant
  return props.messages.filter(m => m.role !== 'tool')
})

// 消息列表引用
const messageListRef = ref(null)

/**
 * 判断当前消息是否处于流式输出状态
 * 只有最后一条 assistant 消息在 streaming 时才显示光标
 */
function isStreamingForMessage(msg) {
  if (!props.streaming || msg.role !== 'assistant') return false
  // 找到 messages 中最后一条 assistant 消息
  const lastAssistant = [...props.messages].reverse().find(m => m.role === 'assistant')
  return lastAssistant === msg
}

// 监听消息变化，自动滚动到底部
watch(
  () => props.messages.length,
  () => {
    nextTick(() => {
      scrollToBottom()
    })
  }
)

// 深度监听消息内容变化（流式更新时也会触发滚动）
watch(
  () => props.messages,
  () => {
    nextTick(() => {
      scrollToBottom()
    })
  },
  { deep: true }
)

/**
 * 滚动到底部
 */
function scrollToBottom() {
  if (messageListRef.value) {
    messageListRef.value.scrollTop = messageListRef.value.scrollHeight
  }
}
</script>

<style scoped>
.chat-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg-app);
}

/* 消息列表 */
.message-list {
  flex: 1;
  overflow-y: auto;
  padding: 24px 0;
}

.message-list::-webkit-scrollbar {
  width: 6px;
}

.message-list::-webkit-scrollbar-track {
  background: transparent;
}

.message-list::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 3px;
}

.message-list::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* 空状态 */
.empty-state {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 40px 24px;
  text-align: center;
  background: var(--bg-app);
}

.empty-icon {
  margin-bottom: 24px;
}

.empty-icon img {
  width: 320px;
  height: auto;
}

.empty-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 8px;
}

.empty-tagline {
  font-size: 14px;
  color: var(--text-muted);
  margin-bottom: 32px;
}

.example-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: center;
  max-width: 540px;
}

.example-chip {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 10px 14px;
  color: var(--text-secondary);
  cursor: default;
  transition: border-color var(--transition), color var(--transition);
}


/* 消息列表 */
.messages {
  max-width: 1200px;
  margin: 0 auto;
}
</style>
