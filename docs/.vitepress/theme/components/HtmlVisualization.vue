<script setup lang="ts">
import { ref, computed, onBeforeUnmount } from 'vue'
import { withBase } from 'vitepress'

const props = defineProps<{
  src: string
  height?: string
  title?: string
}>()

const iframeSrc = computed(() => withBase(props.src))
const minHeight = computed(() => props.height || '400px')

const iframeEl = ref<HTMLIFrameElement | null>(null)
const computedHeight = ref(props.height || '400px')

let resizeObserver: ResizeObserver | null = null

function readHeight(): number {
  const iframe = iframeEl.value
  if (!iframe) return 0
  try {
    const body = iframe.contentDocument?.body
    if (!body) return 0
    // 临时设为 1px，使 min-height: 100vh = 1px，scrollHeight 反映真实内容高度
    const prev = iframe.style.height
    iframe.style.height = '1px'
    const h = body.scrollHeight
    iframe.style.height = prev
    return h
  } catch {
    return 0
  }
}

function updateHeight() {
  const h = readHeight()
  if (h > 0) computedHeight.value = `${h}px`
}

function onLoad() {
  updateHeight()
  const iframe = iframeEl.value
  if (!iframe) return
  try {
    const body = iframe.contentDocument?.body
    if (!body) return
    resizeObserver = new ResizeObserver(updateHeight)
    resizeObserver.observe(body)
  } catch { /* 同源访问失败时静默降级 */ }
}

onBeforeUnmount(() => {
  resizeObserver?.disconnect()
  resizeObserver = null
})
</script>

<template>
  <div class="html-visualization">
    <div v-if="title" class="viz-title">{{ title }}</div>
    <iframe
      ref="iframeEl"
      :src="iframeSrc"
      :style="{ height: computedHeight, minHeight: minHeight }"
      frameborder="0"
      loading="lazy"
      sandbox="allow-scripts allow-same-origin"
      @load="onLoad"
    />
  </div>
</template>

<style scoped>
.html-visualization {
  margin: 16px 0;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  overflow: hidden;
}
.html-visualization iframe {
  width: 100%;
  display: block;
}
.viz-title {
  padding: 8px 16px;
  font-weight: 600;
  background: var(--vp-c-bg-soft);
  border-bottom: 1px solid var(--vp-c-divider);
}
</style>
