---
title: Rope
date: 2025-08-25 24:00:00
updated: 2025-08-25 24:00:00
tags:
  - 机器学习
categories:
  - 机器学习
description: Rope理解
mathjax: true
---
# RoPE

##  维度公式

 维度公式

- 公式：

$$
X.shape = bs \times seq\_length \times hidden\_dim
$$

- 各部分含义：

    - **bs**: batch size（批量大小）。
    - **seq_length**: 序列长度（句子或输入 token 数）。
    - **hidden_dim**: 隐藏维度（每个 token 的表示向量维度）。

所以 **X** 就是 Transformer 输入张量，它的维度是三维的：

- 第一维：批量数  
- 第二维：序列长度  
- 第三维：隐藏向量维度  

---

在 RoPE 中，位置编码就是作用在这个张量上，通常只对 **hidden_dim** 的前一半做 **cos/sin 旋转**，以嵌入位置信息。

##  2. 为什么需要使用RoPE?

绝对位置编码模型只能学习到[第n个词]的位置, **不能很好的表现其相对位置**

可学习的位置embedding,只能在训练过的长度范围内使用。超过最大长度时，位置 embedding 不知道该怎么处理，**无法外推到更长的序列**。

综合可知, 上面的根据公式可知,其两个token就算换位置了,位置矩阵向量还是原来的矩阵向量, 则原来的位置编码会出现外推性不足, 出现了远程衰减性

RoPE的优势

1. 旋转操作把位置信息嵌入向量，**天然地编码相对位置**。
2. 与 Attention 点积契合，能直接影响 $QK^T$ 的相似度计算。
3. 具备 **长序列外推能力**，在训练范围外依然能处理更长文本

因此，RoPE 在保留位置信息的同时，更好地支持长序列建模和远程依赖捕获，已经成为现代大模型的主流选择。

##  3. 公式理解

在标准的自注意力机制中，注意力权重由查询向量（Query）与键向量（Key）之间的点积计算得到，其形式为：

$$
\text{Attention}(Q,K) \propto QK^T
$$

具体而言，若考虑第 $m$ 个 token 的查询向量 $Q_m$ 与第 $N$ 个 token 的键向量 $K_N$，其注意力得分由内积：

$$
\langle Q_m , K_N \rangle
$$

确定。

然而，这一计算过程仅能反映向量间的语义相似性，并未显式建模 token 之间的**相对位置信息**。因此，模型无法区分相邻 token 与远距离 token 的关系强弱，即它并不具备**“相邻词语在语义上应当更相关”**的归纳偏置。这一局限使得原始的注意力机制在捕获序列结构信息时存在不足，从而影响其对长序列中依赖关系的建模能力。

则我们选择使用了RoPE, 下面我们来一步一步进行推理
$$
\langle \text{RoPE}(Q_m), \ \text{RoPE}(K_n) \rangle
$$

$$
\text{RoPE}(Q_m) = 
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\vec{Q}_{m}
$$

$$
\text{RoPE}(K_n) =
\begin{bmatrix}
\cos(m\theta) & -\sin(m\theta) \\\\
\sin(m\theta) & \cos(m\theta)
\end{bmatrix}
\vec{K}_{n}
$$

为什么这样子做呢? 因为我们引入了旋转矩阵
$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\\\
\sin\theta & \cos\theta
\end{bmatrix}
$$
旋转矩阵的性质
**转置等于逆矩阵**

$$
R(\alpha)^{\top} = R(-\alpha) = 
\begin{bmatrix}
\cos\alpha & \sin\alpha \\\\
-\sin\alpha & \cos\alpha
\end{bmatrix}
$$

**旋转可叠加**

$$
R(\alpha + \beta) = R(\alpha) \, R(\beta)
$$
则假设有两个向量,我们求作用旋转矩阵后的内积, 则

设向量 $\vec{x}, \vec{y}$，有：

$$
\begin{aligned}
\langle \, R(\alpha)\vec{x}, \; R(\beta)\vec{y} \, \rangle
&= \vec{x}^{\top} \, R(\alpha)^{\top} \, R(\beta) \, \vec{y} \\\\
&= \vec{x}^{\top} \, R(-\alpha) \, R(\beta) \, \vec{y} \\\\
&= \vec{x}^{\top} \, R(\beta - \alpha) \, \vec{y}
\end{aligned}
$$
根据这个公式后, 我们再回到最初的公式
$$
\langle \text{RoPE}(Q_m), \ \text{RoPE}(K_n) \rangle
$$
具体展开为：

$$
\begin{aligned}
\langle R(m\theta)Q_m, \ R(n\theta)K_n \rangle
&= Q_m^T R(m\theta)^T R(n\theta) K_n \\\\
&= Q_m^T R(-m\theta) R(n\theta) K_n \\\\
&= Q_m^T R((n-m)\theta) K_n
\end{aligned}
$$
则最初的点积公司$\langle Q_m , K_N \rangle$ 变为 
$$
\langle Q_m , K_N \rangle >> Q_m^T R((n-m)\theta) K_n
$$
由此可见，RoPE 的关键性质在于: 经过旋转位置编码后，注意力计算结果不仅依赖于向量本身的相似性，还显式包含了 token 之间的**相对位置差 $(n-m)$**。  

其中$\theta$决定着其外推力, $\theta$就是 RoPE 的“位置频率控制参数”，它决定了旋转速度，也直接影响了 RoPE 在长序列上的外推能力。

**θ 大** → 高频旋转，强调短程关系，外推能力弱。

**θ 小** → 低频旋转，强调长程关系，外推能力强。

##  4 减少远程衰减性的办法

1. 调整 θ 的分布（改变频率范围）

    **减小频率下界**（让高维度对应的 θ 更小） → 增强长程区分度。

    **拉伸频率范围**（修改基数 10000，比如改成更大值） → 延缓远程混叠出现。

2. 混合位置编码

    在 RoPE 基础上，**混合绝对或相对位置编码**，可以让模型同时保留远程与短程信息

​	- RoPE 捕捉相对位置。

​	- 可学习位置 embedding 或 ALiBi（Attention with Linear Biases）补充远程衰减控制。

3. 动态/分段 θ 调整

    不是所有维度 θ 都线性分布，而是做 **非均匀分布**：

    - 一部分维度保持大 θ（短程敏感）。

    - 一部分维度专门分配更小的 θ（远程敏感）。

    类似于给模型增加“短程通道”和“远程通道”。