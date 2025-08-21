---
title: Transformer
date: 2025-08-22 23:00:00
updated: 2025-08-22 23:00:00
tags:
  - 机器学习
categories:
  - 机器学习
description: Transformer整体架构与理解
---
# Transformer

## 1. 背景

Transformer(变换器/转换器)是一种 深度学习模型架构, 在Transformer之前分为四个阶段

1. **基于特征工程的传统方法**

    人工设计特征（TF-IDF、词袋模型 BoW、n-gram，再用 SVM、逻辑回归等分类器), 此方法电脑并不理解语言,只会计算字词出现的次数,单纯的逻辑统计

2. **词向量与分布式表示（2013–2014）**

    **Word2Vec / GloVe（2013-2014）**：学习词向量，把词映射到语义空间), 进步在于学会了词与词之间的语义相似度, 和Transformer缺少了顺序(Order) 和 上下文(Context),这个

3. **RNN / LSTM / GRU 时代（2014 起）重点**

    循环神经网络（RNN）能够按时间顺序处理序列，考虑词语的顺序关系。RNN 在处理时，将前一个词的隐藏状态（hidden state）传递给下一个词，天然包含顺序信息。但其缺点是：序列过长会导致梯度消失或爆炸（数值指数增长/衰减），且难以并行计算。

    LSTM（长短期记忆网络，遗忘门、输入门、输出门）和 GRU（门控循环单元，更新门、重置门）通过“门机制”缓解了长序列建模中梯度消失的问题，使得模型能够捕捉更长的依赖关系。但它们依然存在训练速度慢、并行度低的局限。

    在这一阶段，一个重大突破是 **编码器–解码器（Encoder–Decoder）框架** 的提出（2014）。

    - **编码器（Encoder）**：将输入序列逐步读入，并压缩成一个固定维度的语义向量（context vector）。
    - **解码器（Decoder）**：以该语义向量为输入，逐步生成目标序列，直到遇到终止符 `<EOS>`。
    - **意义**：这一框架首次解决了“输入序列与输出序列长度不一致”的问题.

    另一个重大突破在于 **注意力机制**(2015)的提出

    - 原因: 因为Encoder–Decoder 的局限,当序列数据很长时, 固定向量难以承载所有信息, 编码器最后一个 hidden state难以承载所有信息, 前面的信息可能丢失.信息压缩瓶颈 + 长序列信息丢失问题
    - **Attention**核心思想: 解码器在生成输出的每一个词时，不再只依赖于那个压缩向量. 它可以“动态地”去关注输入序列的不同部分. 
    - **Attention**方法: 权重打分, 加权求和.
    - 这让模型在生成每个词时，都能根据需要去‘对齐’输入句子的不同部分

4. **CNN 在 NLP 中的尝试**

    有些研究用卷积神经网络（CNN）处理文本（比如 TextCNN、ByteNet、ConvS2S）其更擅长局部模式,  拥有语义理解(固定词向量), 并没有进行突破.

**根据上面的阶段可知, Transformer之前主流的序列转导模型的结构是:**

1. 根据RNN/CNN
2. 使用编码器/解码器(Encoder and Decoder)
3. 使用注意力机制增强.

##  2. 问题

NLP在上面的发展中出现了 语义理解, 顺序, 但是并没有更好的效果出现, Google 2017年《Attention Is All You Need》提出了Transformer 的模型架构, 直接用 **自注意力机制（Self-Attention）** 来建模序列关系。

**解决的关键的问题**

1. 长序列导致的梯度消失/爆炸 
1. 并行处理
1. 更好的表达能力
1. 通用架构

## 3.  论文解析

[Attention Is All You Need (arXiv 1706.03762)](https://arxiv.org/pdf/1706.03762)

### **3.1 Abstract**

Transformer 是一种仅基于注意力机制、摒弃循环与卷积的新型模型，在机器翻译任务中显著提升了性能与训练效率，并具备良好的泛化能力。

### **3.2 Introduction**

传统的 RNN、LSTM、GRU 在序列建模和翻译中效果很好，但存在顺序计算无法并行化的问题，尤其在长序列时效率低。

虽然有一些改进方法（如因式分解和条件计算）提升了效率，但顺序依赖的根本瓶颈仍然存在。

注意力机制能建模远距离依赖关系，但之前大多还是和循环网络结合使用。

**Transformer** 完全抛弃循环，只依赖注意力机制，大幅提升并行化能力，并能在较短训练时间内达到当时机器翻译的最佳水平。

### **3.3 Background**

之前的模型（如 Extended Neural GPU、ByteNet、ConvS2S）通过卷积实现并行化，但在处理远距离依赖时仍然效率低下。

自注意力（self-attention）能有效捕捉序列中不同位置的关系，已在阅读理解、摘要和语义推理等任务中取得成功。

记忆网络采用注意力机制替代循环，在一些语言任务中表现良好。

**Transformer 是首个完全依赖自注意力、不使用循环或卷积的模型，大幅提升了建模效率与表达能力。**

###  **3.4 Model Architecture**

![Transformer 架构图](https://img.darkmoonrise.top/myblog/rag%E5%9B%BE%E5%83%8F%E9%A2%86%E5%9F%9F.excalidraw.png)

#### Attention

根据架构来分析整体流程: 以 我爱中国(以512为其语义向量维度) 作为输入语句,按照框架会将语义向量和位置信息向量融合在一起, 现在回过头看这个**向量数据已经拥有了语义信息和位置编码信息**, 现在**缺少上下文的能力**,体现在的模型并不理解 我爱中国 的中是 中国还是中间的意思. 所以下一步要有注意力机制, 而Transformer在注意力机制中也得到了创新:  **自注意力 (Self-Attention)**. 

![第一步](https://img.darkmoonrise.top/myblog/rag%E5%9B%BE%E5%83%8F%E9%A2%86%E5%9F%9F.excalidraw(7).png)




此时 我爱中国 其中这四个字 每个字的融合向量为 1 x 512, 则 这四个字的向量为 4 x 512向量维度.
$$
X = [x_{\text{我}}, x_{\text{爱}}, x_{\text{中}}, x_{\text{国}}] \in \mathbb{R}^{4 \times 512}
$$

那么：

$$
Q = X \cdot W_Q \quad (4 \times 512)(512 \times 512)
$$

$$
K = X \cdot W_K \quad (4 \times 512)(512 \times 512)
$$

$$
V = X \cdot W_V \quad (4 \times 512)(512 \times 512)
$$

这里的 $W_Q, W_K, W_V$ 就是在 **Multi-Head Attention** 层里定义好的参数矩阵。导出Q, K, V 就是为了解决上下文的问题.

Query : 匹配什么信息 , 举例来讲以我爱中国为例, **中** 是不是名词还是什么词? 

Key : 提供什么信息,  是 / 不是 回答信息

Value : 真正携带的信息, 其 "中"的 本义信息

则根据这些信息, 我们需要 计算Q 和 K的相似度, 来告诉 中 和 我爱中国的关系, 它的意思依赖于**国**, 所以中在Query中去问, **"上下文中, 谁和我最相关" **, 则做 **点积相乘 **来计算相似度. 为什么其和**国的相似度高** 是因为  **语义信息 + 位置编码共同作用**

  Query 去和所有 Key 做点积（相似度计算）。  

$$
score = Q \cdot K^T
$$

- **归一化权重**:  
    用 softmax 得到每个词对当前词的权重。  

- **加权求和**:  
    用权重去加权求和所有 Value, 记住Vaclue 代表着文字的本义信息：  
    $$
    Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

以**中**为例子则可以理解
$$
\text{权重}_1 \cdot 我V + \text{权重}_2 \cdot 爱V + \text{权重}_3 \cdot 中V + \text{权重}_4 \cdot 国V 
= \text{语义} + \text{顺序} + \text{上下文中的新向量}
$$
整个架构为并行计算,则我们现在已经达到了完整的具有 语义 + 顺序 + 上下文的 **新向量(依然为 4 x 512)**, 以上为单头注意力(Self attention)的完整过程.

现在我们回顾整个自注意力机制(Self Attention)的**矩阵维度**的变化, 

1. 以“我爱中国”为例，每个词的 embedding 是 512 维，整体语义矩阵为 (4 × 512)。
2. 加入位置编码后，仍为 (4 × 512)，包含语义 + 顺序信息。
3. 再分别与 WQ, WK, WV (均为 512 × 512) 相乘，得到 Q, K, V，维度依旧是 (4 × 512)，
4. 接着计算 Q 与 K 的点积，经过 softmax 得到注意力权重，
5. 再加权 V，得到最终输出，融合了语义 + 顺序 + 上下文, 维度依旧是 (4 × 512)。

| 步骤               | 数学公式                                                     |
| ------------------ | ------------------------------------------------------------ |
| **(1) 输入**       | $X \in \mathbb{R}^{4 \times 512}$                          |
| **(2) 加位置编码** | $X' = X + PE, \quad X' \in \mathbb{R}^{4 \times 512}$      |
| **(3) 线性映射**   | $Q = X'W_Q, \quad K = X'W_K, \quad V = X'W_V$ <br/> $W_Q, W_K, W_V \in \mathbb{R}^{512 \times 512}$ <br/> $Q, K, V \in \mathbb{R}^{4 \times 512}$ |
| **(4) 注意力分数** | $\text{Score} = QK^T \in \mathbb{R}^{4 \times 4}$ <br/> $\alpha = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)$ |
| **(5) 输出**       | $\text{Attention}(Q,K,V) = \alpha V \quad \in \mathbb{R}^{4 \times 512}$ |

论文给我们引出了 多头注意力（Multi-Head Attention, MHA）, 其整体矩阵变化流程为

1. 以“我爱中国”为例，每个词的 embedding 是 512 维，整体语义矩阵为 (4 × 512)。
2. 加入位置编码后，仍为 (4 × 512)，包含语义 + 顺序信息。
3. 再分别与 WQ, WK, WV (均为 512 × 64) 相乘，得到 Q, K, V，维度是 (4 × 64)，有八份
4. 接着计算 Q 与 K 的点积，经过 softmax 得到注意力权重，注意力权重矩阵 **(4 × 4)**.
5. 再加权 V，得到最终输出，融合了语义 + 顺序 + 上下文, 维度为**(4 × 64)**,对所有 8 个头进行相同操作，得到 8 份 **(4 × 64)**
6. 拼接与线性变换,将 8 个头拼接，得到 **(4 × 512)**。
7. 再经过一个线性变换（全连接层），得到最终输出 **(4 × 512)**，包含语义 + 顺序 + 全局上下文。

| 步骤                   | 数学公式                                                     | 矩阵维度                        |
| ---------------------- | ------------------------------------------------------------ | ------------------------------- |
| (1) 输入               | $ X \in \mathbb{R}^{n \times d_{\text{model}}} $           | $ n \times d_{\text{model}} $ |
| (2) 加入位置编码       | $ X' = X + PE $                                            | $ n \times d_{\text{model}} $ |
| (3) 线性映射生成 Q,K,V | $ Q_i = X' W_Q^{(i)} $ <br> $$ K_i = X' W_K^{(i)} $ <br> $$ V_i = X' W_V^{(i)} $ | $ n \times d_k $              |
| (4) 注意力权重         | $ \text{Score}_i = \frac{Q_i K_i^T}{\sqrt{d_k}} $ <br> $ \alpha_i = \text{Softmax}(\text{Score}_i) $ | $ n \times n $                |
| (5) 单头输出           | $ \text{head}_i = \alpha_i V_i $                           | $ n \times d_k $              |
| (6) 多头拼接           | $ \text{Concat}(\text{head}_1,\ldots,\text{head}_h) $      | $ n \times (h \cdot d_k) $    |
| (7) 最终输出           | $ \text{MultiHead}(Q,K,V) = \text{Concat}(\cdot) W^O $     | $ n \times d_{\text{model}} $ |

为什么使用多头注意力（Multi-Head Attention, MHA）:

1. 捕捉不同的特征子空间, 每一个头都代表这一个特折子空间
2. 更好的表示能力, 单头的话则信息都压缩到一个 **权重矩阵 (n×n)** 里，表达能力有限。
3. 缓解信息丢失问题, 单头注意力可能会 **过于专注某些关系**，导致忽视其他信息
4. 提升训练稳定性, **点积不会过大**，更容易训练和收敛。

为什么最后要再经过一个线性层映射出来呢? 

1. **潜空间融合**  存在一个潜空间规合这多组数据，线性层就是学习找到这个空间。
2. **维度对齐**  线性层把拼接后的多头结果从$(h \cdot d_k)$映射回$ n \times d_{\text{model}} $保证与模型整体维度一致。  

3. **信息融合**  线性层学习如何加权组合不同注意力头的输出，把分散的信息整合成统一表示。  

4. **增加模型表达能力**  线性层相当于再投影，能学习更复杂的组合方式，从而提升模型对语义和上下文的表达能力。  

5. **残差连接匹配**  线性层确保输出与输入维度相同，使得残差连接  $ X + \text{MultiHead}(X) $  能够顺利进行。  

为什么我们使用8个64维度呢?

1. 这个由人为来定义,也可以使用其他, 约定俗成
2. 数据矩阵向量同样为 4 x 512, 多头注意力的好处太多了



####  Add & Norm

1. **Add（残差连接）**  
       `Add` 表示残差连接，它将原始输入矩阵向量 \(X\) 与子层（如多头注意力）的输出 \(\text{Sublayer}(X)\) 相加，得到：  

$$
   X' = X + \text{Sublayer}(X)
$$

   这样既保留了原始输入信息，又引入了子层的新特征，从而增强模型的稳定性和可训练性。  

2. **Norm（层归一化）**  
    在 `Add` 之后，会接 `LayerNorm`：  
    $$
    \text{Output} = \text{LayerNorm}(X')
    $$

    归一化保证了数值分布的稳定，避免梯度消失/爆炸，并加速模型收敛。  

为什么使用 LN 不使用 BN呢?

Transformer 用 LayerNorm 而不是 BatchNorm，因为 LN 不依赖 batch 维度，能稳定处理变长序列和小 batch，保证训练与推理阶段的一致性。



####  Feed Forward

在 Transformer 中，每个位置的输出都会经过一个 **前馈全连接网络**（对每个 token 独立计算，位置之间不交互）：  

$$
\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2
$$

- $ W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}} $

- $ W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}} $

- $d_{\text{ff}}$ 一般比 $d_{\text{model}}$ 大，例如 BERT 中：  
  $$d_{\text{model}} = 768, \quad d_{\text{ff}} = 3072$$

也就是说，FFN 先把维度升高（扩展特征）, 非线性变换 , 再降回去（压缩特征）。  

以 我爱中国 (4 x 512)为例: 

| 步骤               | 数学公式                                                     | 矩阵维度            |
| ------------------ | ------------------------------------------------------------ | ------------------- |
| (1) 输入           | $ X \in \mathbb{R}^{4 \times 512} $                        | $ 4 \times 512 $  |
| (2) 多头注意力输出 | $ \text{MHA}(X) $                                          | $ 4 \times 512 $  |
| (3) Add \& Norm    | $ X' = \text{LayerNorm}(X + \text{MHA}(X)) $              | $ 4 \times 512 $  |
| (4) FFN 升维       | $ H = \max(0, X'W_1 + b_1) $ <br> $ W_1 \in \mathbb{R}^{512 \times 2048} $ | $ 4 \times 2048 $ |
| (5) FFN 降维       | $ Y = HW_2 + b_2 $ <br> $ W_2 \in \mathbb{R}^{2048 \times 512} $ | $ 4 \times 512 $  |
| (6) 最终输出       | $ \text{FFN}(X') $                                         | $ 4 \times 512 $  |

每个 token 的 **512 维向量** 先被映射到 **2048 维**，经过 ReLU 激活，再映射回 **512 维**。

以上为 编码器(Encoder) 的全部过程 

####  Dncoder 

对比 Encoder。Decoder 比 Encoder 多了两块：

1. **Masked Multi-Head Self-Attention**（防止看到未来信息）
2. **Encoder-Decoder Attention**（把 Encoder 的输出作为 Key/Value，帮助解码）

接下来还是以我爱中国为例子, 假设我们要生成句子 “I love China”，到目前生成到 “I love”，Decoder 的输入序列长度为 3，每个 token 的 embedding 维度是 512 则矩阵向量为(3 x 512 )。

#### **Masked Multi-Head Self-Attention**

输入 (Input Embedding + Positional Encoding)

- 输入目标序列（如 “I love”），输入为 ["<BOS>", "I", "love"],  <BOS> (Begin of Sentence)：表示序列的开始，相当于一个提示符。 得到 embedding：  

    $$
    X \in \mathbb{R}^{n \times d_{\text{model}}}, \quad n = 3, \ d_{\text{model}} = 512
    $$

- 加上位置编码：  

    $$
    X' = X + PE
    $$

- 此时矩阵向量维度为 3 x 512

Masked Multi-Head Self-Attention

1. 输入 embedding（加了位置编码以后）：  

$$
X' \in \mathbb{R}^{3 \times 512}
$$

​	对应 3 个 token：  

​	- 第 1 个 `<BOS>`  

​	- 第 2 个 `"I"`  

​	- 第 3 个 `"love"`  

2. 计算 Q, K, V

​	对每个 token 生成：  

$$
Q = X' W_Q, \quad K = X' W_K, \quad V = X' W_V
$$

​	假设用 8 个头，每个头的维度：  

$$
d_k = 64
$$

​	那么：  

$$
Q, K, V \in \mathbb{R}^{3 \times 64}
$$

​	（对每个头而言）。拼接后再线性变换，最终维度恢复为：  

$$
3 \times 512
$$

3. 注意力得分 (Score)

​	计算注意力分数矩阵：  

$$
\text{Score} = \frac{QK^T}{\sqrt{d_k}}
$$
4. Mask 处理

普通自注意力：每个词都能看到所有词。  

但在 Decoder 的 **Masked Self-Attention** 中：  

- `<BOS>` 只能看自己  
- `"I"` 可以看 `<BOS>` 和 `"I"`  
- `"love"` 可以看 `<BOS>`、`"I"` 和 `"love"`  
- 未来词被 **mask** 成 \(-\infty\)，Softmax 后概率为 0。  

所以 Mask 后的 **注意力矩阵** 形状仍然是 \(3 \times 3\)，但下三角保留，上三角被屏蔽：  

$$
\alpha =
\begin{bmatrix}
1 & 0 & 0 \\
0.5 & 0.5 & 0 \\
0.33 & 0.33 & 0.34
\end{bmatrix}
$$

（这里只是举例，实际概率由 Softmax 计算得到）。  

5. 加权求和

把注意力权重和 \(V\) 相乘：  

$$
\text{MaskedMHA}(X') = \alpha V
$$

输出维度：  

$$
\mathbb{R}^{3 \times 512}
$$

#### **Encoder-Decoder Attention**

1. 输入和来源

- **Query (Q)**：来自 Decoder 的上一层输出（也就是经过 **Masked Multi-Head Attention + Add&Norm** 后的 ${MaskedMHA}(X')$,$\mathbb{R}^{3 \times 512}$，即 `["<BOS>", "I", "love"]`）。
- **Key (K)、Value (V)**：来自 Encoder 的输出（比如源语言 “我 爱 中国”，经过 Encoder 处理后得到的矩阵，维度是 m×dmodelm \times d_{\text{model}}m×dmodel，这里 m=4m=4m=4，因为有 4 个 token）。

2. 作用

- Decoder 在生成目标语言时，不仅要依赖自身上下文（Masked Self-Attention），还要 **看源语言**。
- 这个机制就是 **Encoder-Decoder Attention**：
    - Q = Decoder 当前生成的上下文（比如已经有 `<BOS>, I, love`）。
    - K、V = Encoder 对源语言的编码表示（比如 “我爱中国” 的上下文信息）。
- 通过计算$ {QK^T}$，Decoder 能找到目标词与源语言中哪些位置对齐。

3. 数学公式

    注意力计算：  
    $$
    \text{EncDecAttention}(X_1, \text{EncOutput}) 
    = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

    - $ Q \in \mathbb{R}^{n \times d_k} $ （来自 Decoder）  
    - $ K, V \in \mathbb{R}^{m \times d_k} $ （来自 Encoder）  
    - 输出维度：  

    $$
    \in \mathbb{R}^{n \times d_{\text{model}}}
    $$

    也就是说，Decoder 的每个 token （`<BOS>`, `I`, `love`）都会去 Encoder 的整个源句子 “我 爱 中国” 上做一次「注意力对齐」。  

4. 直观理解

    - Encoder 输出的是“源语言的理解”：比如 “我爱中国” → 「我=主语，爱=动词，中国=宾语」。
    - Decoder 需要在生成目标语言时，利用这些信息：
        - 当生成 `"I"` 时，Decoder 的 Query 会主要对齐到 Encoder 的 “我”。
        - 当生成 `"love"` 时，Query 会主要对齐到 Encoder 的 “爱”。
        - 当生成 `"China"` 时，Query 会主要对齐到 Encoder 的 “中国”。

    这就实现了 **源语言和目标语言之间的对齐（alignment）**。

#### Linear and Softmax

1. Linear 层

Decoder 最后一层的输出是：  

$$
H \in \mathbb{R}^{n \times d_{\text{model}}}
$$

- $n$：目标序列长度（比如 3，对应 `<BOS>, I, love`）。  
- $d_{\text{model}}$：隐藏维度（比如 512）。  

接下来要做预测：输出词表里到底是哪一个词？  

所以需要一个 **Linear 层（全连接层）**：  

$$
Z = HW^T + b
$$

其中：  

- $ W \in \mathbb{R}^{|V| \times d_{\text{model}}} \quad (|V| = \text{词表大小，比如 } 30,000) $  

- 输出：  $ Z \in \mathbb{R}^{n \times |V|} $

也就是说，Linear 层把 **每个 token 的 512 维隐藏向量**，映射到一个 **词表大小的分数向量**。  

2. Softmax

得到的 $Z$ 只是“分数”，还不是概率。  

通过 **Softmax**：  

$$
P(y_i = w \mid x) = \frac{\exp(Z_{i,w})}{\sum_{v=1}^{|V|} \exp(Z_{i,v})}
$$

- 对于第 $i$ 个 token，Softmax 会把分数转成一个概率分布。  
- 这个概率分布的维度是 $|V|$，所有候选词概率加起来等于 1。  

最终我们就能从中选出最可能的下一个词。  

3. 举例

以 `["<BOS>", "I", "love"]` 为例：  

- **Linear 层**：把 `"love"` 的隐藏向量  
    $$
    h \in \mathbb{R}^{512}
    $$
    映射成  
    $$
    \mathbb{R}^{30000}
    $$
    对应词表里每个词的分数。  

- **Softmax**：把这 30,000 个分数归一化，得到一个概率分布。  

例如：  

- $$ P("China") = 0.65 $$  
- $$ P("dog") = 0.05 $$  
- $$ P("pizza") = 0.01 $$  
- …  

模型选择概率最高的 `"China"` 作为下一个词。  

4. 总结

- **Linear**：把隐藏表示投影到词表空间（维度从 $d_{\text{model}} \to |V|$）。  
- **Softmax**：把分数转成概率分布，用来做下一个词的预测。  



##  4. 时间复杂度

| 层类型                      | 每层计算复杂度             | 顺序操作数 | 最大路径长度     |
| --------------------------- | -------------------------- | ---------- | ---------------- |
| 自注意力 (Self-Attention)   | $$O(n^2 \cdot d)$$         | $$O(1)$$   | $$O(1)$$         |
| 循环网络 (Recurrent)        | $$O(n \cdot d^2)$$         | $$O(n)$$   | $$O(n)$$         |
| 卷积网络 (Convolutional)    | $$O(k \cdot n \cdot d^2)$$ | $$O(1)$$   | $$O(\log_k(n))$$ |
| 限制性自注意力 (restricted) | $$O(r \cdot n \cdot d)$$   | $$O(1)$$   | $$O(n / r)$$     |

**每层计算复杂度 (Complexity per Layer)**：单层运算需要的计算量。

**顺序操作数 (Sequential Operations)**：是否能并行计算，越小越容易并行。

**最大路径长度 (Maximum Path Length)**：信息在序列中传播所需的最长路径。