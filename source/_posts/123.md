---
title: Bayesian Estimation
date: 2025-08-27 15:00:00
updated: 2025-08-27 15:00:00
tags:
  - 机器学习
categories:
  - 数学
description: 概率论
---
# 概率
##  1. 绪论

什么是概率 --- 频率学派 和 贝叶斯学派

假设 抛硬币, 正面概率为 $\theta$ , 反面概率为 $1-\theta$, 其抛硬币的结果为**7次**正面,**3次**反面,求 抛正/负的概率($\theta$/$1 - \theta$)

1.  $\theta$ : 参数			
2. 抛硬币的结果: 数据(x) 

##  2. 频率学派(最大似然估计MLE)

**频率学派:** 1. 参数, 固定的, 客观存在的 2. 数据, 随机的, 抽样获得的

其认为只要数据足够大, 便可以逼近一个**确定的真实**的参数值
$$
{L}(\theta)= \theta^{7}(1 - \theta)^{3}
$$
**“根据最大似然估计（MLE），参数估计值为 θ^=0.7，所以估计的正面概率是 0.7，反面概率是 0.3。**”

##  3. 贝叶斯学派(Bayesian Estimation)

###  3.1 理论

贝叶斯学派: 1. 参数, 随机的, 主观信念的表达 2. 数据, 固定的, 观测到的具体值 3. 先验知识: 在你还没抛硬币之前, 对这枚硬币公平性的直接或者信念, 如 正反为0.5.
$$
\text{后验(参数)} \propto \text{似然(数据)} \times \text{先验}
$$

- **后验** ($\(P(\theta \mid x)\)$) → 参数在看到数据后的分布  
- **似然** ($\(P(x \mid \theta)\)$) → 数据给出的证据  
- **先验** ($\(P(\theta)\)$) → 在没看到数据前对参数的信念    

核心思想, 通过已有的知识和新获得的证据,来推断某个参数取值或事件发生的可能性

参数是随机的, 需要 数据(x)  与 先验知识 结合来得到 正/ 负概率, 下面为推导式,推导贝叶斯公式

从条件概率定义出发：

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \qquad 
P(B \mid A) = \frac{P(A \cap B)}{P(A)}.
$$

把两式右边的 $P(A \cap B)$ 相等联立：

$$
P(A \mid B) P(B) = P(B \mid A) P(A).
$$

令 $A = \theta$ （参数），$B = x$ （数据），得到：

$$
P(\theta \mid x) = \frac{P(x \mid \theta) P(\theta)}{P(x)}.
$$

其中分母：

$$
P(x) = \int P(x \mid \theta) P(\theta)\, d\theta
$$

是**归一化常数**（又叫证据、边际似然），保证 $P(\theta \mid x)$ 的积分为 1。

于是得到常用的“成比例”写法：

$$
P(\theta \mid x) \propto P(x \mid \theta) \times P(\theta).
$$

---

参数
$$
\theta \in [0,1] \quad (\text{正面概率})
$$

数据
$$
x = (k=7 \text{ 次正面}, \; n-k=3 \text{ 次反面})
$$

**似然（数据给的证据）**

忽略二项系数时：
$$
P(x \mid \theta) = \theta^7 (1-\theta)^3
$$

更完整的二项分布形式：
$$
P(x \mid \theta) = \binom{10}{7} \, \theta^7 (1-\theta)^3
$$

**先验（看到数据前对 $\theta$ 的信念）**

常用 **Beta 先验**：
$$
\theta \sim \mathrm{Beta}(\alpha,\beta)
$$

其密度函数为：
$$
P(\theta) = \frac{1}{B(\alpha,\beta)} \, \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

若“完全不了解”，可取均匀先验： 
$$
\theta \sim \mathrm{Beta}(1,1)
$$

**后验（合并得到）**
$$
P(\theta \mid x) \propto P(x \mid \theta) \, P(\theta)
$$

代入具体形式：
$$
P(\theta \mid x) \propto \theta^7(1-\theta)^3 \cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

化简：
$$
P(\theta \mid x) \propto \theta^{(\alpha+7)-1}(1-\theta)^{(\beta+3)-1}
$$

因此：
$$
\theta \mid x \sim \mathrm{Beta}(\alpha+7, \; \beta+3)
$$

**特例：均匀先验**

当 $\alpha=\beta=1$ 时：
$$
\theta \mid x \sim \mathrm{Beta}(8,4)
$$

其后验期望：(正面朝上)
$$
\mathbb{E}[\theta \mid x] = \frac{8}{8+4} = \frac{2}{3} \approx 0.667
$$

###  3.2 优势

1. 小样本数据训练更优
2. 不确定性场景建模, 智能驾驶
3. 对神经网络可解释性的提升,天然支持,先验,后验, 先验的递归更新机制,适合信息逐步积累,及时调整信念的场景

##  4 最大后验估计(MAP)

- **思想**：结合先验 + 数据，选择让 **后验概率** 最大的参数值。  

- **公式**：  
    $$
    \hat{\theta}_{\text{MAP}} 
    = \underset{\theta}{\arg\max} \; P(\theta \mid x) 
    = \underset{\theta}{\arg\max} \; \big[ P(x \mid \theta)\, P(\theta) \big]
    $$

- **与 MLE 的关系**：

    - 如果先验是 **均匀分布**（即不偏向任何参数），MAP 和 MLE 完全一样。  
    - 如果先验有偏好（例如觉得硬币偏公平），MAP 会“折中”数据与先验。  

- **通俗理解**：

    - MLE 是“光看数据”；  
    - MAP 是“数据 + 我的先验直觉（信念）”；  
    - MAP = “我认为最可能的参数值”。  

##  5 对比

| 方法         | 核心思想     | 数学形式                                    | 通俗理解                         |
|--------------|--------------|---------------------------------------------|----------------------------------|
| **MLE**      | 最大似然     | $\arg\max_\theta P(x \mid \theta)$          | 只看数据，哪个参数最可能生成数据 |
| **MAP**      | 最大后验     | $\arg\max_\theta\,[P(x \mid \theta)P(\theta)]$ | 数据 + 先验，选最可能的参数      |
| **贝叶斯估计** | 后验均值/分布 | $\mathbb{E}[\theta \mid x]$ 或整个后验分布     | 不只一个点，而是“参数的全貌”     |
