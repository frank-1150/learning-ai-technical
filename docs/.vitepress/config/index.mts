import { defineConfig } from 'vitepress'
import { shared } from './shared.mts'
import { zh } from './zh.mts'
import { en } from './en.mts'

export default defineConfig({
  ...shared,
  locales: {
    root: { label: '简体中文', ...zh },
    en: { label: 'English', ...en }
  }
})
