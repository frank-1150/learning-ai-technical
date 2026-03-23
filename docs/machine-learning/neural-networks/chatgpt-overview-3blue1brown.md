---
date: 2026-03-18
title: ChatGPT Overview (3Blue1Brown)
description: GPT/Transformer parameter breakdown notes based on the 3Blue1Brown video series
tags: [neural-networks, transformer, gpt, attention, 3blue1brown]
---

# ChatGPT Overview (3Blue1Brown)

## 3Blue1Brown Videos

1. [(Transformers, the tech behind LLMs | Deep Learning Chapter 5)[https://www.youtube.com/watch?v=wjZofJX0v4M]](https://youtu.be/wjZofJX0v4M?si=mwm38F1NFhcwu2ih)
2. [Attention in transformers, step-by-step | Deep Learning Chapter 6](https://youtu.be/eMlx5fFNoYc?si=aqXOSO_hvC3UJWP8)
3. [How might LLMs store facts | Deep Learning Chapter 7 [https://www.youtube.com/watch?v=9-Jl0dxWQs8]](https://youtu.be/9-Jl0dxWQs8?si=gLsXhdHI3G4wRf2y)

## Notes

Breaking down GPT-3's parameters. Out of 175B parameters:

1. Token embedding (mapping text to tokens) accounts for 617 million; the reverse mapping (embedding back to tokens) also uses 617 million.
2. Multi-head attention accounts for 14.4 billion × 4 = 57.6 billion.
3. The multi-layer neural network accounts for the remaining parameters, roughly. This shows that most of the knowledge is actually stored in the MLP weights. Chapter 7 of the video explains how these parameters store information numerically.

![](../../images/Pasted%20image%2020260315212820.png)
