---
title: CS336 学习资料
description: Stanford CS336 "Language Modeling from Scratch" 的学习笔记，从零实现一个语言模型的全过程
tags: [cs336, stanford, language-model, from-scratch]
---

# CS336：Language Modeling from Scratch

[CS336](https://stanford-cs336.github.io/) 是 Stanford 的一门"从零造轮子"的课程：不调 HuggingFace，自己写 tokenizer、自己写 Transformer、自己写 AdamW、自己做分布式训练和推理优化。它的价值不在于教你新算法，而在于逼你把每一个"大家都知道"的环节亲手算一遍、写一遍。

这个系列是我做作业过程中觉得值得单独拿出来分享的部分。

## 笔记列表

| 笔记 | 对应内容 |
|---|---|
| [AdamW 训练需要多少显存和算力](./adamw-memory-and-compute) | Assignment 1 · §4 Training LLM |

## 相关阅读

课程后半部分的推理与分布式内容，与本站 [推理优化与硬件](../inference/nvidia-vera-rubin-lpx) 系列有大量重叠，可以对照着看。
