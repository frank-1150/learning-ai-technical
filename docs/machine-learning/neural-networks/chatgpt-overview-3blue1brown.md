---
title: ChatGPT 概览 (3Blue1Brown)
description: 基于 3Blue1Brown 系列视频的 GPT/Transformer 参数结构笔记
tags: [neural-networks, transformer, gpt, attention, 3blue1brown]
---

# ChatGPT 概览 (3Blue1Brown)

## 3Blue1Brown 的视频

1. [(Transformers, the tech behind LLMs | Deep Learning Chapter 5)[https://www.youtube.com/watch?v=wjZofJX0v4M]](https://youtu.be/wjZofJX0v4M?si=mwm38F1NFhcwu2ih)
2. [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://youtu.be/eMlx5fFNoYc?si=aqXOSO_hvC3UJWP8)
3. [How might LLMs store facts | Deep Learning Chapter 7 [https://www.youtube.com/watch?v=9-Jl0dxWQs8]](https://youtu.be/9-Jl0dxWQs8?si=gLsXhdHI3G4wRf2y)

## 笔记

把GPT-3 的按照参数来拆分。175B 的参数中：

1. 把文字映射成 token 的token embedding 部分占了 617 million；从 embedding 转 token 的部分又有 617 million
2. attention 部分的多头注意力，总共占了 14.4 billion \* 4 = 57.6 billions
3. 多层的神经网络部分占了剩下的部分，大约。说明大部分的知识其实是保存在多成神经网络的参数里。视频的 Chapter 7 部分介绍了这些部分的参数是如何通过数字存储信息的

![](../../images/Pasted%20image%2020260315212820.png)
