---
title: RAGAS
date: 2025-08-12 23:00:00
updated: 2025-08-12 23:00:00
tags:
  - 机器学习
categories:
  - 机器学习
description: RAGAS各项数据指标,计算公式及内涵。
---
# RAGAS

RAGAS 是一个用于评估RAG系统的框架,它允许在不依赖人工注释的情况下,通过一套指标评估检索模块和生成模块的性能机器生成质量 

|   评估数据集 (Dataset)   |       评估指标 (Metric)       |
| :----------------------: | :---------------------------: |
|     Question(问题集)     |     Faithfulness(忠诚度)      |
|     Answer(RAG答案)      | Answer Relevance(答案相关性)  |
| Contexts(检测到的上下文) | Context Relevance(片段相关性) |
| Ground_Truths(标准答案)  |    Context Recall (召回率)    |

## 1. Faithfulness(忠诚度)

衡量生成答案与给定上下文之间的事实一致性. 忠诚单独得分是基于答案和检索到的上下文计算出来的, 答案评分范围在0到1之间,分数越高越好. 
$$
\text{Faithfulness score} = \frac{\text{Number of claims in the generated answer that can be inferred from given context}}{\text{Total number of claims in the generated answer}}
$$
公式解释: 将RAG生成的答案分成N块, 将这些N块分别和 contexts(上下文) 进行检测, 看 contexts 中是否存在这些N块

---

例子: 假设生成回答中有 5 条陈述：

1. 上海市是中国的直辖市 （上下文中有）
2. 上海人口约 2500 万 （上下文中有）
3. 上海是中国的首都 （上下文中没有，而且是错误的）
4. 上海有迪士尼乐园 （上下文中有）
5. 上海面积 9000 平方公里 （上下文中没有）

- 分子：3（第 1、2、4 条）

- 分母：5（所有陈述）

- 公式：
    $$
    \textit{Faithfulness} = \frac{3}{5} = 0.6
    $$

其忠诚度为 0.6 



## 2. Answer Relevance(答案相关性)

答案相关性的评估指标旨在评估生成的答案与给定提示的相关程度。如果答案不完整或包含冗余信息，则会被赋予较低的分数. 这个指标使用问题和答案来计算，其值介于 0 到 1 之间，得分越高表明答案的相关性越好。
$$
AR = \frac{1}{n} \sum_{i=1}^{n} \text{sim}(q, q_i)
$$
其中：  

**q**：用户原本问的问题

**qi**：你回答里的每一句话（或每个要点）

**sim(q,qi)**：把这一句话和原问题做“相似度对比”，看看这句话是不是在围绕问题说事

**n**：你的回答里总共有多少句话（要点）

**结果**：所有句子和问题的相似度取平均值，就是 AR 分数

---

例子: 法国在西欧，它的首都是巴黎。

- 完整回答了两个问题
- 没有多余信息
 - **高度相关**

**打分过程**：

- 句 1：法国在西欧 → 相似度 0.9
- 句 2：首都是巴黎 → 相似度 0.95
- AR = 0.9+0.952=0.925\frac{0.9 + 0.95}{2} = 0.92520.9+0.95=0.925

##  3. Context Relevance(片段相关性)  

上下文精确度衡量上下文中所有相关的真实信息是否被排在了较高的位置。  理想情况下，所有相关的信息块都应该出现在排名的最前面。  这个指标是根据问题和上下文来计算的，数值范围在 0 到 1 之间，分数越高表示精确度越好。

公式：  

$$
\text{Context Precision@k} = \frac{\sum \text{precision}@k}{\text{total number of relevant items in the top K results}}
$$

$$
\text{Precision@k} = \frac{\text{true positives}@k}{\text{true positives}@k + \text{false positives}@k}
$$

其中 \( k \) 是上下文中信息块的总数。

---

例子: 中国在哪里？它的首都是哪？

假设检索系统返回了 5 个上下文片段（按检索排名顺序排列）：

| 排名 |              上下文片段内容              | 是否相关（Relevant?） |
| :--: | :--------------------------------------: | --------------------- |
|  1   |        中国位于亚洲，其首都是北京        | 相关                  |
|  2   | 中国地貌包括草原、沙漠、山脉、湖泊、河流 | 相关                  |
|  3   |        北京是中国的政治和文化中心        | 相关                  |
|  4   |          长城是世界文化遗产之一          | （与问题不直接相关）  |
|  5   |         上海是中国的重要经济中心         | （与问题不直接相关）  |

**Precision@k** 表示在前 k 个结果中，有多少比例是相关的。

- **Precision@1** = 相关数 / k = 1 / 1 = **1.0**
- **Precision@2** = 2 / 2 = **1.0**
- **Precision@3** = 3 / 3 = **1.0**
- **Precision@4** = 3 / 4 = **0.75**
- **Precision@5** = 3 / 5 = **0.6**

Context Precision@k 计算

公式是：
$$
\text{Context Precision@k} =
\frac{\sum_{i=1}^{m} \text{Precision@}k_i}
{\text{相关信息块总数}}
$$

$$
\text{Context Precision@k} =
\frac{1.0 + 1.0 + 1.0}{3} = 1.0
$$

这里：

- 相关信息块总数 = **3**（因为我们有 3 个真正相关的片段）
- 它们出现在排名 **1、2、3**，所以我们只取 **Precision@1、Precision@2、Precision@3** 来计算。

则片段相关性为1.0 

## 4. Context Recall (上下文召回率)

用于衡量检索到的上下文与被视为事实真相的标注答案的一致性程度。  它根据事实真相和检索到的上下文来计算，数值范围在 0 到 1 之间，数值越高表示性能越好。为了从事实真相的答案中估计上下文召回率，需要分析答案中的每个句子是否可以归因于检索到的上下文。  在理想情况下，事实真相答案中的所有句子都应该能够对应到检索到的上下文中。

公式：  

$$
\text{context recall} =
\frac{\left| \text{GT sentences that can be attributed to context} \right|}
{\left| \text{Number of sentences in GT} \right|}
$$

其中：  

- GT（Ground Truths）表示标注的真实答案  
- 分子是标注答案中可以由上下文支持的句子数  
- 分母是标注答案中的总句子数

---

例子: 中国在哪里？它的首都是哪？

**标注答案（Ground Truth, GT）：**

1. 中国位于亚洲。
2. 中国的首都是北京。
3. 北京以紫禁城闻名。

**假设检索到的上下文（Relevant Context）**

- 片段 A： 中国位于亚洲。 正确
- 片段 B： 北京是中国的首都。 正确
- 片段 C： 中国有长城。 错误（和 GT 无直接对应）

**计算步骤**
$$
\text{Context Recall} =
\frac{\text{GT 中可以由上下文支持的句子数}}
{\text{GT 总句子数}}
$$

$$
\text{Context Recall} = \frac{2}{3} \approx 0.67
$$

- 0.67 表示标注答案中有 **67% 的句子** 被检索到的上下文覆盖到了  
- 如果能在上下文中找到支持“北京以紫禁城闻名”的信息，那么分数会提升到 **1.0**

## 5. 总结
<img src="https://img.darkmoonrise.top/myblog/ragas.png" alt="RAGas" style="zoom:33%;" />

### 5.1 检索器

1. **Context precision（上下文精确度）**：评估检索质量。  
2. **Context Recall（上下文召回率）**：衡量检索的完整性。  

### 5.2 生成器

1. **Faithfulness（忠实度）**：衡量生成答案中的幻觉情况。  
2. **Answer Relevance（答案相关性）**：衡量答案对问题的直接性（紧扣问题的核心）。



**最终的RAGAS得分是以上各个指标得分的调和平均值. 简而言之, 这些指标用来综合评估一个系统整体的性能**