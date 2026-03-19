import DefaultTheme from 'vitepress/theme'
import DocDate from './components/DocDate.vue'
import HtmlVisualization from './components/HtmlVisualization.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: DocDate,
  enhanceApp({ app }) {
    app.component('HtmlVisualization', HtmlVisualization)
  }
}
