<template>
  <aside class="sidebar">
    <!-- 侧边栏头部 -->
    <div class="sidebar-header">
      <div class="logo">
        <img class="logo-icon" src="../assets/quanmotor-logo.svg" alt="QuanMotor" />
      </div>
      <button class="new-chat-btn" @click="$emit('new-chat')">
        <span class="icon">+</span>
        新建对话
      </button>
    </div>

    <!-- 搜索框 -->
    <div class="search-box">
      <input
        type="text"
        v-model="searchKeyword"
        placeholder="搜索会话..."
        class="search-input"
      />
    </div>

    <!-- 会话列表 -->
    <div class="session-list">
      <div class="session-list-header">
        <span>历史记录</span>
      </div>

      <div class="session-items">
        <div
          v-for="session in filteredSessions"
          :key="session.thread_id"
          class="session-item"
          :class="{ active: session.thread_id === currentThreadId }"
          @click="$emit('select-session', session.thread_id)"
        >
          <div class="session-info">
            <span class="session-title">{{ session.title }}</span>
            <span class="session-time">{{ formatTime(session.updated_at) }}</span>
          </div>
          <button
            class="delete-btn"
            @click.stop="handleDelete(session.thread_id)"
            title="删除会话"
          >
            ×
          </button>
        </div>

        <!-- 空状态 -->
        <div v-if="sessions.length === 0" class="empty-state">
          <p>暂无会话记录</p>
          <p class="hint">开始一个新对话吧</p>
        </div>
      </div>
    </div>

    <!-- 侧边栏底部 -->
    <div class="sidebar-footer">
      <div class="footer-info">
        <span class="version">v1.0.0</span>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { ref, computed } from 'vue'

// Props 定义
const props = defineProps({
  sessions: {
    type: Array,
    default: () => []
  },
  currentThreadId: {
    type: String,
    default: null
  }
})

// Emits 定义
const emit = defineEmits(['select-session', 'new-chat', 'delete-session'])

// 搜索关键词
const searchKeyword = ref('')

// 过滤后的会话列表
const filteredSessions = computed(() => {
  if (!searchKeyword.value) {
    return props.sessions
  }
  return props.sessions.filter(session =>
    session.title.toLowerCase().includes(searchKeyword.value.toLowerCase())
  )
})

/**
 * 格式化时间
 */
function formatTime(timestamp) {
  if (!timestamp) return ''

  const date = new Date(timestamp)
  const now = new Date()
  const diff = now - date

  // 一天内显示相对时间
  if (diff < 24 * 60 * 60 * 1000) {
    const hours = Math.floor(diff / (60 * 60 * 1000))
    if (hours < 1) {
      const minutes = Math.floor(diff / (60 * 1000))
      return minutes < 1 ? '刚刚' : `${minutes} 分钟前`
    }
    return `${hours} 小时前`
  }

  // 超过一天显示日期
  return `${date.getMonth() + 1}/${date.getDate()}`
}

/**
 * 处理删除会话
 */
function handleDelete(threadId) {
  if (confirm('确定要删除这个会话吗？')) {
    emit('delete-session', threadId)
  }
}
</script>

<style scoped>
.sidebar {
  width: 280px;
  height: 100vh;
  background: var(--bg-sunken);
  color: var(--text);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  border-right: 1px solid var(--border);
  position: relative;
}

/* 侧边栏头部 */
.sidebar-header {
  padding: 20px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-sunken);
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 20px;
}

.logo-icon {
  height: 48px;
  width: auto;
}

.logo-text {
  font-size: 20px;
  font-weight: 600;
  color: var(--text);
  letter-spacing: 0.5px;
}

.new-chat-btn {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  padding: 12px 20px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color var(--transition);
}

.new-chat-btn:hover {
  background: var(--accent-hover);
}

.new-chat-btn:active {
  background: var(--accent-hover);
}

.new-chat-btn .icon {
  font-size: 20px;
  font-weight: 300;
}

/* 搜索框 */
.search-box {
  padding: 16px;
}

.search-input {
  width: 100%;
  padding: 12px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  color: var(--text);
  font-size: 14px;
  outline: none;
  transition: border-color var(--transition), box-shadow var(--transition);
}

.search-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.search-input::placeholder {
  color: var(--text-muted);
}

/* 会话列表 */
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}

.session-list::-webkit-scrollbar {
  width: 4px;
}

.session-list::-webkit-scrollbar-track {
  background: transparent;
}

.session-list::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 2px;
}

.session-list::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

.session-list-header {
  padding: 12px 16px 8px;
  font-size: 12px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

.session-items {
  padding: 0 12px;
}

.session-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  margin-bottom: 4px;
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: background-color var(--transition), box-shadow var(--transition);
  border: 1px solid transparent;
}

.session-item:hover {
  background: var(--bg-hover);
}

.session-item.active {
  background: var(--accent-soft);
  box-shadow: inset 2px 0 0 var(--accent);
  color: var(--text);
}

.session-info {
  flex: 1;
  overflow: hidden;
}

.session-title {
  display: block;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--text);
  font-weight: 500;
}

.session-time {
  display: block;
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 3px;
}

.session-item.active .session-time {
  color: var(--text-secondary);
}

.delete-btn {
  opacity: 0;
  padding: 6px 10px;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-muted);
  font-size: 18px;
  line-height: 1;
  cursor: pointer;
  transition: background-color var(--transition), color var(--transition), opacity var(--transition);
}

.session-item:hover .delete-btn {
  opacity: 1;
}

.delete-btn:hover {
  background: var(--danger-soft);
  color: var(--danger);
}

/* 空状态 */
.empty-state {
  padding: 60px 24px;
  text-align: center;
  color: var(--text-muted);
}

.empty-state p {
  margin-bottom: 10px;
}

.empty-state .hint {
  font-size: 13px;
  color: var(--text-muted);
}

/* 侧边栏底部 */
.sidebar-footer {
  padding: 16px;
  border-top: 1px solid var(--border);
  background: var(--bg-sunken);
}

.footer-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.version {
  font-size: 12px;
  color: var(--text-muted);
  letter-spacing: 0.5px;
}
</style>
