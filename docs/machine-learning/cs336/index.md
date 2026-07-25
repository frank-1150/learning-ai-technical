---
title: CS336 Notes
description: Study notes from Stanford CS336 "Language Modeling from Scratch" — building a language model end to end
tags: [cs336, stanford, language-model, from-scratch]
---

# CS336: Language Modeling from Scratch

[CS336](https://stanford-cs336.github.io/) is a build-it-yourself course: no HuggingFace, you write the tokenizer, the Transformer, AdamW, the distributed training loop, and the inference stack yourself. Its value isn't new algorithms — it's being forced to actually compute and implement every step everyone claims to already understand.

This series collects the parts of the assignments I found worth writing up on their own.

## Notes

| Note | Covers |
|---|---|
| [How much memory and compute does training GPT-2 XL take?](./adamw-memory-and-compute) | Assignment 1 · §4 Training LLM |

## Related

The inference and distributed sections of the course overlap heavily with the [Inference & Hardware](../inference/nvidia-vera-rubin-lpx) series on this site.
