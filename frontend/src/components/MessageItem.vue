<template>
  <!-- ============================================================ -->
  <!-- 用户消息 - 右侧气泡 -->
  <!-- ============================================================ -->
  <div v-if="message.role === 'user'" class="message-item message-user">
    <div class="message-content-wrapper">
      <div class="message-content">
        <div class="content-text">{{ message.content }}</div>
      </div>
    </div>
    <div class="message-avatar">
      <div class="avatar user-avatar">
        <span class="avatar-label">你</span>
      </div>
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- AI 助手消息 - 左侧，带头部标签栏 -->
  <!-- ============================================================ -->
  <div v-else-if="message.role === 'assistant'" class="message-item message-assistant" :class="{ streaming: isStreaming }">
    <div class="message-avatar">
      <div class="avatar assistant-avatar" :class="getAvatarClass(message.source)">
        <svg v-if="!message.source || message.source === 'main'" viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M8 0c.6 3.9 2.9 6.6 8 8-5.1 1.4-7.4 4.1-8 8-.6-3.9-2.9-6.6-8-8 5.1-1.4 7.4-4.1 8-8z"/></svg>
        <svg v-else viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M8 0c.6 3.9 2.9 6.6 8 8-5.1 1.4-7.4 4.1-8 8-.6-3.9-2.9-6.6-8-8 5.1-1.4 7.4-4.1 8-8z"/></svg>
      </div>
    </div>
    <div class="message-content-wrapper">
      <!-- 角色标签栏 -->
      <div class="msg-label-bar" :class="getLabelBarClass(message.source)">
        <span class="msg-role-badge">{{ getRoleLabel(message.source) }}</span>
        <span v-if="message.source && message.source !== 'main'" class="msg-source-badge">
          {{ getSourceName(message.source) }}
        </span>
      </div>
      <!-- 消息正文 -->
      <div class="message-content">
        <MarkdownRenderer :content="message.content" />
        <span v-if="isStreaming && message.content" class="typing-cursor">▋</span>
      </div>
    </div>
  </div>

  <!-- ============================================================ -->
  <!-- 工具调用消息 - 左侧，带 Tool 标签栏 -->
  <!-- ============================================================ -->
  <div v-else-if="message.role === 'tool'" class="message-item message-tool-inline">
    <div class="message-avatar">
      <div class="avatar tool-avatar">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor" aria-hidden="true"><path d="M14.7 4.2a4.5 4.5 0 0 1-5.9 5.9L4 14.9A1.8 1.8 0 1 1 1.4 12.3l4.8-4.8a4.5 4.5 0 0 1 5.9-5.9L9.6 4.1l2.5 2.5 2.6-2.4z"/></svg>
      </div>
    </div>
    <div class="message-content-wrapper">
      <!-- Tool 标签栏 -->
      <div class="msg-label-bar tool-label-bar">
        <span class="msg-role-badge tool-role-badge">Tool</span>
        <span class="msg-tool-name-badge">{{ formatToolName(message.tool_name) }}</span>
        <span v-if="message.source && message.source !== 'main'" class="msg-source-badge tool-source-badge">
          {{ getSourceName(message.source) }}
        </span>
        <span v-if="message.tool_status === 'calling' && !hasToolContent" class="tool-status-badge calling">执行中</span>
        <span v-else-if="hasToolContent" class="tool-status-badge done">完成</span>
      </div>

      <!-- 参数（可折叠） -->
      <div v-if="message.args" class="tool-inline-section">
        <div class="tool-section-header" @click="toggleSection('args')" :class="{ collapsed: !expandedSections.includes('args') }">
          <svg class="chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4,6 8,10 12,6"/></svg>
          <span class="section-label">参数</span>
        </div>
        <pre v-if="expandedSections.includes('args')" class="tool-inline-code">{{ formatArgs(message.args) }}</pre>
      </div>

      <!-- 结果文本（可折叠） -->
      <div v-if="message.text" class="tool-inline-section">
        <div class="tool-section-header" @click="toggleSection('result')" :class="{ collapsed: !expandedSections.includes('result') }">
          <svg class="chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4,6 8,10 12,6"/></svg>
          <span class="section-label">结果</span>
        </div>
        <div v-if="expandedSections.includes('result')" class="tool-inline-result">
          <MarkdownRenderer :content="cleanText(message.text)" />
        </div>
      </div>

      <!-- 结果图片 -->
      <div v-if="message.images && message.images.length > 0" class="tool-inline-images">
        <div
          v-for="(img, imgIdx) in message.images"
          :key="imgIdx"
          class="tool-image-wrapper"
        >
          <img
            :src="img"
            :alt="`图表 ${imgIdx + 1}`"
            class="tool-result-image"
            loading="lazy"
            style="max-width:100%;max-height:360px;width:auto;height:auto;object-fit:contain;border-radius:8px;cursor:pointer;"
            @click="previewImage(img)"
          />
        </div>
      </div>

      <!-- 等待中动画 -->
      <div v-if="message.tool_status === 'calling' && !hasToolContent" class="tool-waiting">
        <span class="waiting-dot"></span>
        <span class="waiting-dot"></span>
        <span class="waiting-dot"></span>
        <span class="waiting-text">执行中...</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import MarkdownRenderer from './MarkdownRenderer.vue'

const props = defineProps({
  message: { type: Object, required: true },
  isStreaming: { type: Boolean, default: false }
})

const expandedSections = ref(['result'])

const hasToolContent = computed(() => {
  return !!(props.message.text || (props.message.images && props.message.images.length > 0))
})

function toggleSection(name) {
  const idx = expandedSections.value.indexOf(name)
  if (idx === -1) { expandedSections.value.push(name) }
  else { expandedSections.value.splice(idx, 1) }
}

function previewImage(src) { window.open(src, '_blank') }

// ============================================================
// AI 消息标签
// ============================================================

/** AI 消息的角色标签文字 */
function getRoleLabel(source) {
  if (!source || source === 'main') return 'AI'
  return getSourceName(source)  // 如「图表代理」
}

/** AI 消息标签栏的 CSS class */
function getLabelBarClass(source) {
  if (!source || source === 'main') return 'label-bar-main'
  return 'label-bar-sub'
}

/** AI 消息头像的 CSS class */
function getAvatarClass(source) {
  if (!source || source === 'main') return 'avatar-main'
  return 'avatar-sub'
}

// ============================================================
// 工具相关
// ============================================================

function formatToolName(name) {
  if (!name) return '未知工具'
  const nameMap = {
    'task': '委派子代理 (task)',
    'generate_column_chart': '生成柱状图',
    'generate_bar_chart': '生成柱状图',
    'generate_line_chart': '生成折线图',
    'generate_pie_chart': '生成饼图',
    'generate_area_chart': '生成面积图',
    'generate_scatter_chart': '生成散点图',
    'web_search': '网络搜索',
    'read_file': '读取文件',
    'write_file': '写入文件',
    'execute_python': '执行 Python 代码',
  }
  return nameMap[name] || name
}

function getSourceName(source) {
  const sourceMap = {
    'main': '主助手',
    'chart-agent': '图表代理',
    'researcher': '研究代理',
    'model-agent': '模型代理',
    'general': '通用代理'
  }
  return sourceMap[source] || source
}

function formatArgs(args) {
  if (!args) return ''
  try { return JSON.stringify(JSON.parse(args), null, 2) }
  catch { return args }
}

function cleanText(text) {
  if (!text) return ''
  let cleaned = text
  cleaned = cleaned.replace(/,?\s*'id':\s*'[^']*'/g, '')
  cleaned = cleaned.replace(/,?\s*"id":\s*"[^"]*"/g, '')
  return cleaned
}
</script>

<style scoped>
/* ============================================================ */
/* 基础布局 */
/* ============================================================ */
.message-item {
  display: flex;
  gap: 12px;
  padding: 12px 24px;
  max-width: 100%;
}

.message-item.streaming {
  box-shadow: inset 2px 0 0 var(--accent);
}

/* ============================================================ */
/* 角色标签栏（AI 和 Tool 共用） */
/* ============================================================ */
.msg-label-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  padding: 6px 0;
  flex-wrap: wrap;
}

.msg-role-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 12px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.3px;
}

/* 主 Agent AI 标签 */
.label-bar-main .msg-role-badge {
  background: var(--accent-soft);
  color: var(--accent);
}

/* 子 Agent AI 标签 */
.label-bar-sub .msg-role-badge {
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid var(--accent-border);
}

/* Tool 标签 */
.tool-label-bar .tool-role-badge {
  background: var(--warning-soft);
  color: var(--warning);
}

.msg-source-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-weight: 600;
}

.label-bar-sub .msg-source-badge {
  background: var(--accent-soft);
  color: var(--accent);
  border: 1px solid var(--accent-border);
}

.tool-source-badge {
  background: var(--warning-soft);
  color: var(--warning);
}

.msg-tool-name-badge {
  display: inline-flex;
  align-items: center;
  padding: 3px 12px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 700;
  color: var(--warning);
  background: var(--warning-soft);
  font-family: 'Monaco', 'Menlo', 'JetBrains Mono', monospace;
}

.tool-status-badge {
  margin-left: auto;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-weight: 600;
}

.tool-status-badge.calling {
  background: var(--warning-soft);
  color: var(--warning);
}

.tool-status-badge.done {
  background: var(--success-soft);
  color: var(--success);
}

/* ============================================================ */
/* 用户消息（右侧） */
/* ============================================================ */
.message-user {
  justify-content: flex-end;
}

.message-user .message-content-wrapper {
  max-width: 65%;
  text-align: right;
}

.message-user .message-content {
  background: var(--accent-soft);
  border: 1px solid var(--accent-border);
  color: var(--text);
  border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
  padding: 14px 18px;
}

.message-user .content-text {
  color: var(--text);
  font-size: 15px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

/* ============================================================ */
/* AI 助手消息（左侧） */
/* ============================================================ */
.message-assistant { justify-content: flex-start; }

.message-assistant .message-content-wrapper {
  max-width: 80%;
  overflow: hidden;
}

.message-assistant .message-content {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 4px var(--radius-lg) var(--radius-lg) var(--radius-lg);
  padding: 14px 18px;
  line-height: 1.7;
  overflow: hidden;
  font-size: 15px;
}

/* ============================================================ */
/* 工具调用消息（左侧） */
/* ============================================================ */
.message-tool-inline { justify-content: flex-start; }

.message-tool-inline .message-content-wrapper {
  max-width: 85%;
  background: var(--warning-soft);
  border: 1px solid var(--warning-border);
  border-radius: 4px var(--radius-lg) var(--radius-lg) var(--radius-lg);
  padding: 14px 18px;
  overflow: hidden;
}

/* 工具内容区域 */
.tool-inline-section { margin-top: 8px; }

.tool-section-header {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  padding: 6px 0;
  user-select: none;
  color: var(--warning);
  font-size: 13px;
  font-weight: 600;
}

.chevron { transition: transform var(--transition); }
.collapsed .chevron { transform: rotate(-90deg); }

.section-label {
  letter-spacing: 0.5px;
  font-size: 12px;
}

.tool-inline-code {
  background: var(--warning-soft);
  border: 1px solid var(--warning-border);
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 13px;
  font-family: 'Monaco', 'Menlo', 'JetBrains Mono', monospace;
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 4px 0 0 20px;
  color: var(--text-secondary);
  line-height: 1.6;
  max-height: 200px;
  overflow-y: auto;
}

.tool-inline-result {
  margin: 4px 0 0 20px;
  padding: 10px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 14px;
  color: var(--text);
  line-height: 1.6;
  overflow: hidden;
}

/* 图片展示 */
.tool-inline-images {
  margin-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow: hidden;
}

.tool-image-wrapper {
  display: flex;
  justify-content: center;
  background: var(--bg-surface);
  border-radius: 10px;
  padding: 8px;
  overflow: hidden;
}

.tool-result-image {
  max-width: 100% !important;
  max-height: 360px;
  width: auto;
  height: auto;
  border-radius: 8px;
  cursor: pointer;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
  object-fit: contain;
  background: var(--bg-sunken);
  display: block;
}

.tool-result-image:hover {
  transform: scale(1.02);
  box-shadow: var(--shadow-md);
}

/* 等待动画 */
.tool-waiting {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 0;
  margin-top: 4px;
}

.waiting-dot {
  width: 8px;
  height: 8px;
  background: var(--warning);
  border-radius: 50%;
  animation: bounce 1.4s infinite ease-in-out both;
}

.waiting-dot:nth-child(1) { animation-delay: -0.32s; }
.waiting-dot:nth-child(2) { animation-delay: -0.16s; }

.waiting-text { margin-left: 8px; font-size: 13px; color: var(--warning); }

/* ============================================================ */
/* 通用样式 */
/* ============================================================ */
.message-avatar { flex-shrink: 0; }

.avatar {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
}

.avatar-label {
  font-size: 12px;
  font-weight: 600;
}

.user-avatar {
  background: var(--text);
  color: #fff;
}

.avatar-main {
  background: var(--accent);
  color: #fff;
}

.avatar-sub {
  background: var(--accent-soft);
  color: var(--accent);
}

.tool-avatar {
  background: var(--warning-soft);
  color: var(--warning);
}

.typing-cursor {
  display: inline-block;
  animation: blink 1s infinite;
  color: var(--accent);
  margin-left: 3px;
  font-weight: bold;
}

/* ============================================================ */
/* 动画 */
/* ============================================================ */
@keyframes blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.4); opacity: 0.4; }
  40% { transform: scale(1); opacity: 1; }
}
</style>
