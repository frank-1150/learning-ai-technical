import DefaultTheme from 'vitepress/theme'
import HtmlVisualization from './components/HtmlVisualization.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HtmlVisualization', HtmlVisualization)
  }
}
