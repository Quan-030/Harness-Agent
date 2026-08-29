<template>
  <div v-if="visible && toolCalls.length > 0" class="tool-call-panel">
    <!-- 面板头部 -->
    <div class="panel-header" @click="togglePanel" :class="{ collapsed: !isExpanded }">
      <div class="header-left">
        <span class="panel-title">工具调用详情</span>
        <span class="tool-count">({{ toolCalls.length }})</span>
      </div>
      <svg class="chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4,6 8,10 12,6"/></svg>
    </div>

    <!-- 工具列表 -->
    <div v-show="isExpanded" class="tool-list">
      <div
        v-for="(tool, index) in toolCalls"
        :key="tool.id || index"
        class="tool-item"
        :class="{ expanded: expandedTools.includes(tool.id || index) }"
      >
        <!-- 工具头部 -->
        <div class="tool-header" @click="toggleExpand(tool.id || index)" :class="{ collapsed: !expandedTools.includes(tool.id || index) }">
          <span class="tool-name">{{ tool.name }}</span>
          <span v-if="tool.source && tool.source !== 'main'" class="tool-source">
            ({{ formatSource(tool.source) }})
          </span>
          <svg class="chevron" viewBox="0 0 16 16" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="4,6 8,10 12,6"/></svg>
        </div>

        <!-- 工具详情 -->
        <div v-if="expandedTools.includes(tool.id || index)" class="tool-details">
          <!-- 参数 -->
          <div v-if="tool.args" class="detail-section">
            <div class="detail-label">参数:</div>
            <pre class="detail-content">{{ formatArgs(tool.args) }}</pre>
          </div>

          <!-- 结果 -->
          <div v-if="tool.result" class="detail-section">
            <div class="detail-label">结果:</div>
            <pre class="detail-content result">{{ tool.result }}</pre>
          </div>

          <!-- 无数据提示 -->
          <div v-if="!tool.args && !tool.result" class="no-data">
            暂无详细信息
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

/**
 * 工具调用面板组件
 *
 * 显示 AI 助手调用工具的详细信息
 */

const props = defineProps({
  toolCalls: {
    type: Array,
    default: () => []
  }
})

// 控制面板显示/隐藏
const visible = ref(true)
// 面板是否展开（控制整个工具列表的折叠）
const isExpanded = ref(true)
// 展开的工具列表
const expandedTools = ref([])

/**
 * 切换面板展开/收起状态
 */
function togglePanel() {
  isExpanded.value = !isExpanded.value
}

/**
 * 切换工具展开/收起状态
 */
function toggleExpand(toolId) {
  const index = expandedTools.value.indexOf(toolId)
  if (index === -1) {
    expandedTools.value.push(toolId)
  } else {
    expandedTools.value.splice(index, 1)
  }
}

/**
 * 格式化来源名称（中文）
 */
function formatSource(source) {
  const sourceMap = {
    'chart-agent': '图表代理',
    'researcher': '研究代理',
    'model-agent': '模型代理',
    'general': '通用代理',
    'main': '主助手'
  }
  return sourceMap[source] || source
}

/**
 * 格式化参数（JSON 格式化）
 */
function formatArgs(args) {
  if (!args) return '无'

  try {
    // 尝试解析为 JSON 并格式化
    const parsed = JSON.parse(args)
    return JSON.stringify(parsed, null, 2)
  } catch {
    // 如果不是 JSON，直接返回原始字符串
    return args
  }
}
</script>

<style scoped>
.tool-call-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  margin: 16px 24px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

/* 面板头部 */
.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 14px 18px;
  background: var(--bg-sunken);
  cursor: pointer;
  border-bottom: 1px solid var(--border);
  transition: background-color var(--transition);
}

.panel-header:hover {
  background: var(--bg-hover);
}

.header-left {
  display: flex;
  align-items: center;
  gap: 8px;
}

.panel-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--text);
}

.tool-count {
  font-size: 13px;
  color: var(--text-muted);
}

/* 面板头部的折叠图标 */
.panel-header .chevron {
  color: var(--text-secondary);
  flex-shrink: 0;
}

/* 工具列表 */
.tool-list {
  max-height: 400px;
  overflow-y: auto;
}

.tool-list::-webkit-scrollbar {
  width: 4px;
}

.tool-list::-webkit-scrollbar-track {
  background: transparent;
}

.tool-list::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 2px;
}

.tool-list::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

.tool-item {
  border-bottom: 1px solid var(--border);
}

.tool-item:last-child {
  border-bottom: none;
}

/* 工具头部 */
.tool-header {
  display: flex;
  align-items: center;
  padding: 14px 18px;
  cursor: pointer;
  gap: 10px;
  transition: background-color var(--transition);
}

.tool-header:hover {
  background: var(--bg-sunken);
}

.tool-name {
  font-weight: 600;
  font-size: 13px;
  font-family: var(--font-mono);
  color: var(--text);
}

.tool-source {
  font-size: 12px;
  color: var(--text-muted);
}

.tool-header .chevron {
  margin-left: auto;
  color: var(--text-secondary);
  flex-shrink: 0;
}

/* SVG chevron rotation */
.chevron {
  transition: transform var(--transition);
}

.collapsed .chevron {
  transform: rotate(-90deg);
}

/* 工具详情 */
.tool-details {
  padding: 0 18px 18px;
  background: var(--bg-sunken);
}

.detail-section {
  margin-bottom: 12px;
}

.detail-section:last-child {
  margin-bottom: 0;
}

.detail-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 6px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}

.detail-content {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  padding: 12px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-family: var(--font-mono);
  overflow-x: auto;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0;
  line-height: 1.6;
  color: var(--text);
}

.detail-content.result {
  border-left: 3px solid var(--accent);
}

.no-data {
  font-size: 13px;
  color: var(--text-muted);
  font-style: italic;
}
</style>
