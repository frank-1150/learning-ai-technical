---
date: 2026-04-17
title: "多模态大模型：ViT、CLIP 与 BLIP-2 如何让语言模型看见图像"
description: "从 Vision Transformer 把图片当 Token 开始，到 CLIP 用对比学习对齐文本和图像，再到 BLIP-2 用 Q-Former 桥接冻结的视觉编码器和 LLM——多模态 LLM 的三块基石。"
tags: [多模态, ViT, CLIP, BLIP-2, Vision Transformer]
---

# 多模态大模型：ViT、CLIP 与 BLIP-2 如何让语言模型看见图像

> 本文对应原书 **第 9 章 Multimodal Large Language Models**，覆盖：Vision Transformer、CLIP 文图对齐、BLIP-2 架构、图像描述与多模态对话。

"Large Language Model" 里那个 **Language** 限定词，从一开始就藏着一个不安的缺口：人类交流从来不只在文字里发生——我们指着照片说话，看图表里找规律，靠面部表情判断语气。把 LLM 局限在纯文本，就像让一个能说会道的人蒙上眼睛再打电话。

这一章要回答的问题就是：**怎么给语言模型"加一双眼睛"？** 答案分三步：先用 **Vision Transformer (ViT)** 把图像变成 Transformer 能吃的 token；再用 **CLIP** 的对比学习把文字和图像对齐到同一个向量空间；最后用 **BLIP-2** 的 Q-Former 把冻结的视觉编码器和冻结的 LLM 桥接起来，让一个已经会"读"的模型学会"看"。

## 多模态 LLM 的演进脉络

从单模态到真正的多模态，行业走过了三段关键路径：

| 阶段 | 代表技术 | 核心突破 | 成本 |
|------|---------|---------|------|
| **第一步：把图像 Transformer 化** | ViT (2020) | 证明 Transformer Encoder 不需要卷积也能在视觉任务上打败 CNN | 训练需要 JFT-300M 级数据 |
| **第二步：文图对齐** | CLIP (2021) | 用 4 亿对图文做对比学习，文本和图像落在同一个向量空间 | 400M 图文对，数百卡训练数周 |
| **第三步：复用现成 LLM** | BLIP-2 (2023) | 冻结视觉编码器和 LLM，只训 100M 参数的 Q-Former 做桥接 | 消费级硬件即可跑推理 |

这条路径有个非常巧妙的共同底色：**每一步都尽可能复用上一步的成果**。ViT 用回了 Transformer Encoder；CLIP 的图像塔直接拿 ViT；BLIP-2 更极致——连 CLIP 和现成 LLM 都原样冻住，只训练中间最薄的一层适配器。这种"积木式"工程哲学也预示了 2024 年后 LLaVA、GPT-4V、Gemini 的主流架构路线。

![多模态 LLM 支持的模态类型](/book-notes/hands-on-llm/images/multimodal-similarity-matrix.png)

## Vision Transformer：把图片当 Token 处理

### 为什么 Transformer 能用在图像上？

这是一个反直觉的起点。过去十年，计算机视觉是 CNN（卷积神经网络）的天下——卷积核天然适合处理网格化像素，有平移不变性、局部感受野等先验优势。而 Transformer 的 Self-Attention 是**全局**的、**无局部偏置**的，看上去跟图像格格不入。

Dosovitskiy 等人在 2020 年发表的 *An Image is Worth 16x16 Words* 给出了一个极简答案：**别再把图像当像素矩阵，把它当成一串"视觉单词"**。

::: tip 核心洞察
Transformer 不在乎输入是"文字 token"还是"图像 patch"。只要你能把输入变成一个个带位置信息的向量，Encoder 的 Multi-Head Attention 就能一视同仁地建模它们之间的全局关系。
:::

### Patch Embedding：图像的 "Tokenization"

ViT 的核心操作就像是"图像的分词"：

1. **切块**：把 224×224 的输入图像切成若干个 16×16 的小方块（patch），得到 $14 \times 14 = 196$ 个 patch
2. **展平 + 线性投影**：每个 $16 \times 16 \times 3 = 768$ 维的 patch，通过一个线性层映射成一个 $D$ 维的 embedding（比如 $D=768$）
3. **加位置编码**：给每个 patch embedding 加上可学习的位置编码，告诉模型 "你是第几行第几列的那一块"
4. **加 [CLASS] token**：参照 BERT，在序列前面拼一个 `[CLASS]` token，Encoder 最后这个位置的输出就是整张图的表示

![ViT 完整架构：从 patch 切分到 encoder 输出](/book-notes/hands-on-llm/images/multimodal-vit.png)

书中把这个过程类比成文本的 tokenization 再贴切不过——只不过 patch 不能像文字 token 那样复用固定词表（因为几乎没有两张图像会共享完全相同的 patch），所以必须用一个**可学习的线性投影**把每个 patch 直接映射成 embedding。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/vit-patching.html" height="520px" title="ViT Patch 切分交互演示" />

### ViT vs CNN：何时选哪个？

| 维度 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 强（平移不变性、局部性） | 弱（全靠数据学出来） |
| 数据量需求 | 中等数据即可 | 需要大量数据（ImageNet-21k 以上） |
| 全局建模能力 | 较弱（靠堆叠层扩大感受野） | 强（Self-Attention 天然全局） |
| 计算复杂度 | $O(n)$ 对序列长度线性 | $O(n^2)$ 对 patch 数平方 |
| 迁移能力 | 需要针对任务微调 | 预训练后 few-shot 能力强 |

结论：**小数据场景 CNN 仍然占优**，但在大规模预训练 + 下游微调的范式下，ViT 展现了更强的"涌现"潜力，也是后续 CLIP / BLIP-2 选择它作图像编码器的主要原因。

## CLIP：把文字和图像拉进同一个向量空间

### 问题：embedding 跨模态的前提

有了 ViT 能把图像编成 embedding，下一个问题是：**图像 embedding 和文本 embedding 怎么比较？** 如果它们分别在不同的向量空间，那"一张猫的图片"和文字"a photo of a cat"就没有任何可比性。

CLIP（Contrastive Language-Image Pre-training）的答案直截了当：**同时训练两个塔，强迫它们的输出落在同一个向量空间**。

### 双塔 + 对比损失

CLIP 的架构极简：

- **Text Encoder**：标准 Transformer Encoder，输入文本 token，输出 `[EOS]` 位置的 embedding
- **Image Encoder**：ViT（早期也用过 ResNet），输入图像 patch，输出 `[CLASS]` 位置的 embedding
- 两个塔的输出都被投影到同一个 $d$ 维空间（比如 $d=512$）

![CLIP 训练的三步流程](/book-notes/hands-on-llm/images/multimodal-clip.png)

训练时的目标函数是 **对比损失 (InfoNCE)**：给定一个 batch 的 $N$ 对 `(image, caption)`，构造一个 $N \times N$ 的相似度矩阵 $S$，其中 $S_{ij} = \cos(\text{img}_i, \text{txt}_j)$。损失要求：

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii} / \tau)}{\sum_{j=1}^{N} \exp(S_{ij} / \tau)}
$$

翻译成人话：**对角线（正样本对）要高，非对角线（负样本对）要低**。$\tau$ 是温度参数，控制分布的 sharp 程度。

::: info 为什么对比学习这么好用？
单独看一条 (image, caption) pair，模型只知道"它们匹配"，学不出"什么是好"。但放在 batch 里跟 $N-1$ 个错误样本一起对比，模型必须学会"为什么 text_A 匹配 image_A，而不是 image_B、image_C...",才能把对角线 push up、其他位置 push down——这就强迫 embedding 学到更细粒度的区分度。
:::

OpenAI 的原版 CLIP 用 4 亿对从互联网爬取的图文训练，这个数据规模直接塞给了 CLIP 一整套 "世界常识"——它知道什么是巴黎铁塔、什么是边牧、什么是 van Gogh 的星夜。

<HtmlVisualization src="/book-notes/hands-on-llm/visualizations/clip-contrastive.html" height="560px" title="CLIP 对比学习相似度矩阵" />

### Zero-Shot 分类：CLIP 最惊艳的用法

训练完的 CLIP 最神奇的能力是**不需要任何分类头、不需要微调，就能分类任意类别**。做法是：

1. 把每个类别名写成一句话模板：`"a photo of a {label}"`
2. 把这些模板编码为文本 embedding
3. 把待分类图像编码为图像 embedding
4. 跟所有文本 embedding 算余弦相似度，取最高的那个作为预测类别

```python
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

# 准备候选类别
labels = ["cat", "dog", "car", "plane", "boat"]
prompts = [f"a photo of a {label}" for label in labels]

# 文本 embedding
text_inputs = clip_tokenizer(prompts, return_tensors="pt", padding=True)
text_embeds = model.get_text_features(**text_inputs)
text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

# 图像 embedding
image_input = clip_processor(images=image, return_tensors="pt")
image_embeds = model.get_image_features(**image_input)
image_embeds /= image_embeds.norm(dim=-1, keepdim=True)

# 相似度打分
similarity = (image_embeds @ text_embeds.T).squeeze(0)
predicted_label = labels[similarity.argmax().item()]
```

### CLIP 的下游应用

| 应用 | 做法 | 典型场景 |
|------|------|---------|
| **图文检索** | 把图库编码一次，查询文本编码后做最近邻搜索 | 图库搜索、无障碍描述 |
| **Zero-shot 分类** | 把类别写成模板文本，与图像比相似度 | 长尾类别、无标注数据的冷启动 |
| **图像聚类** | 用图像 embedding 做 k-means/HDBSCAN | 图库去重、相似图发现 |
| **文生图的语义对齐** | Stable Diffusion 用 CLIP 文本编码器作为条件输入 | 把文字 prompt 翻译成扩散模型能理解的表示 |
| **作为下游多模态模型的视觉编码器** | BLIP-2、LLaVA 等都用 CLIP ViT 作图像侧 encoder | 所有现代视觉-语言模型 |

书中特别推荐了 **OpenCLIP**——OpenAI 原版 CLIP 的开源复现，在更大数据集（LAION-5B）上训练，性能往往超越原版。

### 预处理 CLIP 输入

CLIP 对输入有明确要求：

- **文本**：CLIPTokenizer 加 `<|startoftext|>` / `<|endoftext|>` 特殊符号（注意没有 `[CLS]`，因为 CLIP 用 `[EOS]` 位置作整句表示）
- **图像**：CLIPProcessor 会把任意尺寸的图像 resize 到 **224×224**，做 ImageNet 均值方差归一化

```python
from urllib.request import urlopen
from PIL import Image

image = Image.open(urlopen("https://.../puppy.png")).convert("RGB")
inputs = clip_processor(text=None, images=image, return_tensors="pt")
print(inputs["pixel_values"].shape)  # torch.Size([1, 3, 224, 224])
```

::: warning 图像被挤压的坑
所有非方形图像都会被强制 resize 到 224×224，宽高比严重失衡（比如 520×492 的全景图）时可能出现明显失真。对于细粒度任务，建议先做合理 crop 再喂给 CLIP。
:::

## BLIP-2：让现成的 LLM 学会"看"

### 问题：如何给 LLM 装上眼睛？

CLIP 能把图像和文本对齐到同一个空间，但它毕竟是一个**判别式 embedding 模型**——能打分、能检索，但不会"生成"文字描述。

如果想要一个能看图说话、能回答图片相关问题的模型，最直白的思路是**从头训一个联合多模态模型**。但这成本太高：需要数十亿图文对、数百张 A100、几周训练。

BLIP-2 (Salesforce, 2023) 给出了一个优雅的替代方案：

::: tip BLIP-2 的核心思想
**冻结现成的 Image Encoder (ViT)，冻结现成的 LLM，只训练一个轻量的"桥接器"** —— Querying Transformer (Q-Former)，约 100M 参数。

这意味着我们**复用了 CLIP 已经学到的视觉知识**，也**复用了 LLM 已经学到的语言能力**，只需让桥接器学会"如何把视觉特征翻译成 LLM 能读懂的 soft prompt"。
:::

### Q-Former：桥接视觉与语言

Q-Former 内部其实是两个共享 Self-Attention 层的小 Transformer：

- **Image Transformer**：和冻结的 ViT 交互，通过 Cross-Attention 提取视觉特征
- **Text Transformer**：和文本交互（训练时用 caption，推理时接 LLM）

这两个 Transformer 共享 Self-Attention 权重，但各自有独立的 FFN。关键的"可学习 query"——一组固定数量（比如 32 个）的可训练 embedding——在 Image Transformer 里通过 Cross-Attention 从 ViT 的输出中"抽取"图像信息，最后输出 32 个视觉特征向量。

![BLIP-2 完整架构：冻结的 ViT + 可训练的 Q-Former + 冻结的 LLM](/book-notes/hands-on-llm/images/multimodal-blip2.png)

### 两阶段训练

**Stage 1 — Vision-Language Representation Learning**：只训 Q-Former（ViT 冻结），三个联合目标：

| 任务 | 做法 | 目的 |
|------|------|------|
| **Image-Text Contrastive (ITC)** | 类似 CLIP，拉近匹配对 | 学表示 |
| **Image-Text Matching (ITM)** | 二分类：这对图文匹配吗？ | 学细粒度对齐 |
| **Image-grounded Text Generation (ITG)** | 看图生成 caption | 学生成能力 |

**Stage 2 — Vision-to-Language Generative Learning**：冻结 ViT 和 Q-Former，加一个**线性投影层**把 Q-Former 的 32 个输出投影到 LLM 的 embedding 维度，然后把这些投影向量作为**soft visual prompt**拼到 LLM 的输入前面。训练时只优化投影层（LLM 也冻结）。

最终效果：LLM 拿到的"视觉上下文"和平常拿到的"文本上下文"完全同构，都是一串 embedding，它可以用原有的生成能力直接往后续写文字。

### 为什么 BLIP-2 这么高效？

总训练参数量只有约 100M（vs 从头训数十亿）。关键是它把"学视觉"和"学语言"这两个昂贵的环节彻底**外包**给了现成的预训练模型，只为最后 1% 的"跨模态对齐"付费。

## 实际应用：从图像描述到多模态对话

### 加载 BLIP-2

```python
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

这里 `blip2-opt-2.7b` 指的是以 **OPT-2.7B** 为 LLM 骨干的 BLIP-2 变体（还有 FlanT5 版本）。

### Use Case 1：Image Captioning

最简单的用法，给一张图，让模型自动生成描述：

```python
from urllib.request import urlopen
from PIL import Image

car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/car.png"
image = Image.open(urlopen(car_path)).convert("RGB")

# 不给 text prompt，模型默认做 captioning
inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=20)
caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
print(caption)
# -> "an orange supercar driving on the road at sunset"
```

![BLIP-2 给跑车生成的描述](/book-notes/hands-on-llm/images/multimodal-captioning.png)

这个描述几乎完美——颜色、类型、场景、时间段都抓到了。书中还拿罗夏墨迹测试试了一下，模型给出 "a black and white ink drawing of a bat"，也非常合理。

### Use Case 2：Visual Question Answering

给图 + 一个问题，让模型回答：

```python
prompt = "Question: Write down what you see in this picture. Answer:"
inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
answer = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# -> "A sports car driving on the road at sunset"
```

注意这里的关键：**给 BLIP-2 加了文本 prompt 之后，它会接着 prompt 续写**，而不是像无 prompt 时那样直接输出 caption。这就是"soft prompt 模式"在起作用——图像被编码成 soft prompt 拼在文本 prompt 前面，LLM 根据整段上下文生成答案。

### Use Case 3：多轮多模态对话

把历史问答拼进 prompt，就得到一个"能看图聊天"的 chatbot：

```python
prompt = (
    "Question: Write down what you see in this picture. "
    "Answer: A sports car driving on the road at sunset. "
    "Question: What would it cost me to drive that car? Answer:"
)
inputs = blip_processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
generated_ids = model.generate(**inputs, max_new_tokens=30)
answer = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
# -> "$1,000,000"
```

书中用 `ipywidgets` 搭了一个小 notebook chatbot，可以连续问：

```
USER: Write down what you see in this picture.
BLIP-2: A sports car driving on the road at sunset

USER: What would it cost me to drive that car?
BLIP-2: $1,000,000

USER: Why that much money?
BLIP-2: Because it's a sports car.

USER: Why are sports cars expensive?
BLIP-2: Because they're fast.
```

这种"连续对话"的能力说明 BLIP-2 不仅能看图，还能维持上下文、进行常识推理。

### 预处理细节

- **图像**：任何尺寸都会被 resize 成 224×224 正方形，所以对宽高比失衡的图像要小心
- **文本**：BLIP-2（OPT 版本）用 `GPT2TokenizerFast`，特殊 token 是 `</s>` 作 BOS/EOS；注意 tokenizer 输出里的 `Ġ` 前缀是"空格"的内部表示
- **拼接顺序**：processor 会自动把视觉 soft prompt 放到文本 token 前面，形成 `[visual_tokens] + [text_tokens]`

## CLIP vs BLIP-2：什么时候用哪个？

| 维度 | CLIP | BLIP-2 |
|------|------|--------|
| 类型 | 判别式 embedding 模型 | 生成式 VLM |
| 输出 | 向量（用于检索、分类） | 自然语言文本 |
| 典型任务 | 图文检索、zero-shot 分类、作为 encoder | Captioning、VQA、多模态对话 |
| 训练成本 | 4 亿对图文，高 | 复用预训练，100M 可训参数，中 |
| 推理延迟 | 低（只过两个 Encoder） | 较高（生成式解码） |
| 常见用法 | 做"视觉侧的 embedding 底座" | 做"能说话的视觉助手" |

一句话：**需要快速相似度比较选 CLIP；需要自然语言输出选 BLIP-2**。而且 BLIP-2 里的 Image Encoder 本身就是 CLIP 的 ViT——两者是承接关系，不是替代关系。

## 展望：从 BLIP-2 到 LLaVA、GPT-4V、Gemini

BLIP-2 只是开端。2023 年以后出现了大量基于相同哲学的工作：

- **LLaVA**：用更简单的 MLP 替代 Q-Former，直接把 CLIP 视觉特征投影到 LLM 输入空间，配合 GPT-4 生成的指令跟随数据做 SFT
- **Idefics 2**：基于 Mistral 7B，做了大量架构与数据消融
- **GPT-4V / GPT-4o**：闭源，但大致路线相同——强视觉编码器 + 强 LLM + 跨模态对齐
- **Gemini**：从一开始就做原生多模态训练（而非后期对齐），据称在细粒度视觉推理上更强

所有这些模型的共同范式都来自 BLIP-2 揭示的核心洞察：**不要从零训练多模态模型，而是用最轻的桥接把已有的强模型连起来**。

## 小结

| 技术 | 解决的问题 | 核心机制 | 典型产出 |
|------|-----------|---------|---------|
| **ViT** | Transformer 怎么处理图像 | 把图像切成 16×16 patch 当 token 用 | 统一了视觉和语言的架构范式 |
| **CLIP** | 如何比较图像和文本 | 双塔 + 对比学习，4 亿图文对 | 同一向量空间的文图 embedding、zero-shot 分类 |
| **BLIP-2** | 如何让 LLM 看图说话 | 冻结 ViT + 冻结 LLM + 可训 Q-Former | Image captioning、VQA、多模态对话 |

理解这三块基石，就理解了 2024 年所有主流多模态 LLM（LLaVA / GPT-4V / Gemini / Claude 3）的底层设计逻辑——它们只是在这条路线上换骨干、扩数据、加指令微调。

下一章（Ch 10）将进入"训练与微调"环节，学习如何从零训一个 Embedding 模型——届时对比学习这套"把向量空间掰直"的方法论会再次登场。
