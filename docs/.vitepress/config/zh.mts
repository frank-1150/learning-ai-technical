import { defineConfig, type DefaultTheme } from 'vitepress'

export const zh = defineConfig({
  lang: 'zh-CN',
  description: 'AI 技术学习笔记',

  themeConfig: {
    nav: nav(),
    sidebar: sidebar(),

    editLink: {
      pattern: 'https://github.com/frank-1150/learning-ai-technical/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页面'
    },

    footer: {
      message: '基于 MIT 许可发布',
      copyright: `版权所有 © 2026-present`
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    lastUpdated: {
      text: '最后更新于',
      formatOptions: { dateStyle: 'short', timeStyle: 'medium' }
    },

    returnToTopLabel: '回到顶部',
    sidebarMenuLabel: '菜单',
    darkModeSwitchLabel: '主题',
    lightModeSwitchTitle: '切换到浅色模式',
    darkModeSwitchTitle: '切换到深色模式'
  }
})

function nav(): DefaultTheme.NavItem[] {
  return [
    { text: '首页', link: '/' },
    {
      text: 'AI 应用',
      items: [
        { text: '智能体的原理和操控', link: '/ai-applications/agents/' },
        { text: '我如何使用 AI？', link: '/ai-applications/how-i-use-ai/' }
      ]
    },
    {
      text: '机器学习',
      items: [
        { text: '神经网络', link: '/machine-learning/neural-networks/' },
        { text: '推理优化与硬件', link: '/machine-learning/inference/nvidia-vera-rubin-lpx' }
      ]
    }
  ]
}

function sidebar(): DefaultTheme.Sidebar {
  return {
    '/ai-applications/': [
      {
        text: 'AI 应用',
        items: [
          { text: '概览', link: '/ai-applications/' },
          {
            text: '智能体的原理和操控',
            collapsed: false,
            items: [
              { text: '介绍', link: '/ai-applications/agents/' },
              { text: '智能体的核心：两个循环', link: '/ai-applications/agents/agent-loop' }
            ]
          },
          {
            text: '我如何使用 AI？',
            collapsed: false,
            items: [
              { text: '介绍', link: '/ai-applications/how-i-use-ai/' }
            ]
          }
        ]
      }
    ],
    '/machine-learning/': [
      {
        text: '机器学习',
        items: [
          { text: '概览', link: '/machine-learning/' },
          {
            text: '神经网络',
            collapsed: false,
            items: [
              { text: '介绍', link: '/machine-learning/neural-networks/' },
              { text: 'ChatGPT 概览 (3Blue1Brown)', link: '/machine-learning/neural-networks/chatgpt-overview-3blue1brown' }
            ]
          },
          {
            text: '推理优化与硬件',
            collapsed: false,
            items: [
              { text: 'NVIDIA Vera Rubin + LPX', link: '/machine-learning/inference/nvidia-vera-rubin-lpx' }
            ]
          },
          {
            text: "Let's Build GPT (Karpathy)",
            collapsed: false,
            items: [
              { text: '视频概览', link: '/machine-learning/build-gpt-karpathy/' },
              { text: 'PyTorch 基础操作', link: '/machine-learning/build-gpt-karpathy/pytorch-basics' }
            ]
          }
        ]
      }
    ]
  }
}
