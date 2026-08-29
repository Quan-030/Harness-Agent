<template>
  <div class="input-area">
    <div class="input-container">
      <!-- 工具调用开关 -->
      <div class="tool-toggle">
        <label class="toggle-label">
          <input
            type="checkbox"
            :checked="showToolCalls"
            @change="emit('toggle-tool-calls', $event.target.checked)"
            class="toggle-checkbox"
          />
          <span class="toggle-switch"></span>
          <span class="toggle-text">显示工具调用</span>
        </label>
      </div>

      <!-- 输入框 -->
      <div class="input-wrapper" :class="{ focused: isFocused, streaming: streaming }">
        <textarea
          ref="textareaRef"
          v-model="inputText"
          @focus="isFocused = true"
          @blur="isFocused = false"
          @keydown.enter.exact.prevent="handleSend"
          @input="autoResize"
          placeholder="输入消息，Enter 发送..."
          rows="1"
          :disabled="streaming"
          class="input-textarea"
        ></textarea>
        <!-- 发送按钮：流式时变为停止按钮 -->
        <button
          v-if="!streaming"
          class="send-btn"
          @click="handleSend"
          :disabled="!inputText.trim()"
        >
          <span>发送</span>
        </button>
        <button
          v-else
          class="stop-btn"
          @click="handleStop"
        >
          <span class="stop-icon">■</span>
          <span>停止</span>
        </button>
      </div>

      <!-- 提示信息 -->
      <div class="input-hint">
        <span v-if="!streaming">按 Enter 发送，Shift + Enter 换行</span>
        <span v-else class="streaming-hint">正在生成回复...</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'

/**
 * 输入区域组件
 *
 * 提供消息输入功能：
 * - 正常状态：显示"发送"按钮
 * - 流式输出时：输入框禁用，按钮变为"停止"
 */

const props = defineProps({
  streaming: {
    type: Boolean,
    default: false
  },
  showToolCalls: {
    type: Boolean,
    default: true
  }
})

const emit = defineEmits(['send', 'stop', 'toggle-tool-calls'])

// 输入文本
const inputText = ref('')
// 是否获取焦点
const isFocused = ref(false)
// textarea 引用
const textareaRef = ref(null)

/**
 * 发送消息
 */
function handleSend() {
  const text = inputText.value.trim()
  if (!text) return

  emit('send', text)
  inputText.value = ''

  nextTick(() => {
    if (textareaRef.value) {
      textareaRef.value.style.height = 'auto'
    }
  })
}

/**
 * 停止对话
 */
function handleStop() {
  emit('stop')
}

/**
 * 自动调整 textarea 高度
 */
function autoResize() {
  const textarea = textareaRef.value
  if (!textarea) return

  textarea.style.height = 'auto'
  textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
}
</script>

<style scoped>
.input-area {
  padding: 16px 24px 24px;
  background: var(--bg-surface);
  border-top: 1px solid var(--border);
}

.input-container {
  max-width: 1100px;
  margin: 0 auto;
}

/* 工具调用开关 */
.tool-toggle {
  margin-bottom: 12px;
}

.toggle-label {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  cursor: pointer;
  user-select: none;
}

.toggle-checkbox {
  display: none;
}

.toggle-switch {
  position: relative;
  width: 44px;
  height: 24px;
  background: var(--border-strong);
  border-radius: 12px;
  transition: background-color var(--transition);
}

.toggle-switch::after {
  content: '';
  position: absolute;
  top: 2px;
  left: 2px;
  width: 18px;
  height: 18px;
  background: var(--bg-surface);
  border-radius: 50%;
  transition: transform var(--transition);
  box-shadow: var(--shadow-sm);
}

.toggle-checkbox:checked + .toggle-switch {
  background: var(--accent);
}

.toggle-checkbox:checked + .toggle-switch::after {
  transform: translateX(20px);
}

.toggle-text {
  font-size: 13px;
  color: var(--text-secondary);
  font-weight: 500;
}

/* 输入框包装 */
.input-wrapper {
  display: flex;
  align-items: flex-end;
  gap: 12px;
  padding: 14px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  transition: border-color var(--transition), box-shadow var(--transition);
}

.input-wrapper.focused {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}

.input-wrapper.streaming {
  background: var(--bg-sunken);
  border-color: var(--border-strong);
}

/* 输入框 */
.input-textarea {
  flex: 1;
  border: none;
  background: transparent;
  font-size: 15px;
  line-height: 1.6;
  resize: none;
  outline: none;
  min-height: 24px;
  max-height: 200px;
  color: var(--text);
  font-weight: 500;
}

.input-textarea::placeholder {
  color: var(--text-muted);
}

.input-textarea:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

/* 发送按钮 */
.send-btn {
  flex-shrink: 0;
  padding: 10px 20px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background-color var(--transition);
}

.send-btn:hover:not(:disabled) {
  background: var(--accent-hover);
}

.send-btn:active:not(:disabled) {
  background: var(--accent-hover);
}

.send-btn:disabled {
  background: var(--border);
  color: var(--text-muted);
  cursor: not-allowed;
}

/* 停止按钮 */
.stop-btn {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 10px 18px;
  background: var(--danger);
  color: #fff;
  border: none;
  border-radius: var(--radius-md);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: opacity var(--transition);
}

.stop-btn:hover {
  opacity: 0.9;
}

.stop-icon {
  font-size: 12px;
}

/* 提示信息 */
.input-hint {
  margin-top: 10px;
  text-align: center;
}

.input-hint span {
  font-size: 12px;
  color: var(--text-muted);
}

.streaming-hint {
  color: var(--danger) !important;
  animation: blink 1.5s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
</style>
