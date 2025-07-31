---
title: PCA(主成分分析)推导与代码
date: 2025-07-31
tags:
  - 机器学习
categories:
  - 数学
---

# PCA(主成分分析)

## 1. 前置知识

### 1.1 数据的线性变换

1. 数据的拉伸
    $$
    D =
    \left[
    \begin{matrix}
    x_1 & x_2 & x_3 & x_4 \\\\
    y_1 & y_2 & y_3 & y_4
    \end{matrix}
    \right]
    $$

    $$
    S =
    \begin{bmatrix}
    2 & 0 \\\\
    0 & 1
    \end{bmatrix}
    $$

    $$
    \begin{aligned}
    SD &=
    \begin{bmatrix}
    2 & 0 \\\\
    0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    x_1 & x_2 & x_3 & x_4 \\\\
    y_1 & y_2 & y_3 & y_4
    \end{bmatrix} \\\\
    &=
    \begin{bmatrix}
    2x_1 & 2x_2 & 2x_3 & 2x_4 \\\\
    y_1 & y_2 & y_3 & y_4
    \end{bmatrix}
    \end{aligned}
    $$


2. 数据的旋转
    $$
    D =
    \begin{bmatrix}
    x_1 & x_2 & x_3 & x_4 \\\\
    y_1 & y_2 & y_3 & y_4
    \end{bmatrix}
    $$

    $$
    R =
    \begin{bmatrix}
    \cos(\theta) & -\sin(\theta) \\\\
    \sin(\theta) & \cos(\theta)
    \end{bmatrix}
    $$

    $$
    RD =
    \begin{bmatrix}
    \cos(\theta) & -\sin(\theta) \\\\
    \sin(\theta) & \cos(\theta)
    \end{bmatrix}
    \begin{bmatrix}
    x_1 & x_2 & x_3 & x_4 \\\\
    y_1 & y_2 & y_3 & y_4
    \end{bmatrix}
    $$

3. 白数据: 原始数据

4. 协方差矩阵


    $$
    \text{协方差:} \quad
    \operatorname{cov}(x, y) =
    \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n - 1}
    $$
    ​	**协方差表示的是 两个变量在变化的变化的过程中是同方向变化 ? 还是反方向变化? 同向 或 反向程度如何? **
    $$
    x \uparrow \ \rightarrow\ y \uparrow \ \rightarrow\ \operatorname{cov}(x, y) > 0
    $$
    ​	**当两个变量同时增大(呈正相关趋势) 时, 他们的协方差大于 0**

5. 协方差矩阵


    $$
    C =
    \begin{bmatrix}
    \operatorname{cov}(x,x) & \operatorname{cov}(x,y) \\\\
    \operatorname{cov}(x,y) & \operatorname{cov}(y,y)
    \end{bmatrix}
    \quad \text{对称阵, 对角线是方差}
    $$

## 2. 是什么

PCA本质上是降维, 降维的同时尽可能防止数据失真, 去噪点和增速

## 3.  最大方差主成分推导过程（PCA）

1. 将数据只保留一个轴, 找到数据分布最分散的方向(方差最大), 作为主成分(坐标轴) 数据中心化

     

2. 找到方差最大方向(数据最多且分散),  **拉伸(S)**决定了方差最大的方向是 横 还是 纵 , **旋转(R)** 决定了方差最大的方向的角度 

    > 为什么数据需要拉伸?  
    >
    > ​	为了调整不同方向的方差比例,让重要方向的**特征更突出**, 不重要方向的影响减弱.

    > 为什么旋转代表了 方差最大方向的角度呢?
    >
    > ​	旋转的作用是把 **坐标轴转到数据的主方向上** , 而数据的 **主方向**就是方差最大的方向. 则 我们应该找到 R

    

3. 找到方差最大方向这个问题就变成了 ---> 怎么求 **旋转R ** , 则 **协方差矩阵的特征向量就是 R**  

    根据协方差矩阵  和  协方差

$$
C =
\begin{bmatrix}
\operatorname{cov}(x,x) & \operatorname{cov}(x,y) \\\\
\operatorname{cov}(x,y) & \operatorname{cov}(y,y)
\end{bmatrix}
\quad \text{对称阵, 对角线是方差}
$$


$$
\text{协方差公式:} \quad
\operatorname{cov}(x, y) =
\frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{n - 1}
$$

​			中 xˉ 和 yˉ 为 x 和 y 的 样本均值(数据所有点**最平衡的位置**), 我们需要xˉ 和 yˉ其为0,

​			这样子就可以  **去掉位置偏移的影响** **协方差计算更纯粹**  **方便矩阵计算**  则转化为推导过程
$$
c = 
\begin{bmatrix}
\frac{\sum_{i=1}^n x_i^2}{n-1} & \frac{\sum_{i=1}^n x_i y_i}{n-1} \\
\frac{\sum_{i=1}^n x_i y_i}{n-1} & \frac{\sum_{i=1}^n y_i^2}{n-1}
\end{bmatrix}
$$

$$
x_1 \cdot x_1 + x_2 \cdot x_2 + \cdots + x_n \cdot x_n 
= \sum_{i=1}^n x_i^2
$$

$$
x_1 \cdot y_1 + x_2 \cdot y_2 + \cdots + x_n \cdot y_n 
= \sum_{i=1}^n x_i y_i
$$

$$
\frac{1}{n-1}
\begin{bmatrix}
\sum x_i^2 & \sum x_i y_i \\
\sum x_i y_i & \sum y_i^2
\end{bmatrix}
$$

$$
= \frac{1}{n-1}
\begin{bmatrix}
x_1 & x_2 & x_3 & x_4 \\
y_1 & y_2 & y_3 & y_4
\end{bmatrix}
\begin{bmatrix}
x_1 & y_1 \\
x_2 & y_2 \\
x_3 & y_3 \\
x_4 & y_4
\end{bmatrix}
$$

$$
\mathbf{C} = \frac{1}{n-1} \mathbf{X} \mathbf{X}^T
$$

​	                         	**把原本逐个元素计算的协方差公式，改写成一个简洁的矩阵乘法表达式**。

​		**推导“数据经过旋转（R）+缩放（S）后的协方差矩阵”怎么和原始数据的协方差矩阵联系起来**

​		**原始协方差矩阵：**

$$
C = \frac{1}{n-1} D D^T
$$

$$
D' = R S D (R:旋转, s: 拉伸, 数据施加线性变换)
$$

​		**变换后数据的协方差:**

$$
C' = \frac{1}{n-1} D' {D'}^T
$$

$$
= \frac{1}{n-1} R S D (R S D)^T
$$

$$
= \frac{1}{n-1} R S D D^T S^T R^T
$$

$$
\frac{1}{n-1} D D^T = C
$$

$$
C' = R S C S^T R^T
$$

​		**数据是白化后**

$$
C = \frac{1}{n-1} D D^T = 
\begin{bmatrix}
1 & 0 \\
0 & 1
\end{bmatrix}
$$

$$
C' = R S S^T R^T
$$

​		**设计**
$$
L = S S^T
$$

$$
R S S^T R^T = R L R^T
$$

$$
R^T R = I \quad \Rightarrow \quad R^{-1} = R^T
$$

$$
R L R^T = R L R^{-1}
$$

$$
C' = R L R^{-1}
$$

 

​	有了上面的式子 则我们可知**C' (原坐标系下的协方差)**是已知的 , 其中 **R 代表了(未知的正交矩阵, 特征向量矩阵 , 坐标轴方向)** , **L代表了(未知的对角矩阵, 特征值矩阵, 坐标轴方向的方差)** 接下来 做**特征分解**: 

​	**从协方差矩阵的分解公式开始  **
$$
C' = R \, L \, R^{-1}
$$
​	**对 C' 做特征分解（Eigendecomposition）**  

​	**拉格朗日乘子法（Lagrange Multipliers** 推导出来 
$$
C' \, v_i = \lambda_i \, v_i
$$
​	**排序特征值（从大到小）**  
$$
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n
$$
​	**最大主成分线（First Principal Component）  **

​			**方向：**  
$$
v_{\text{max}} = v_1
$$
​			**方差：**  
$$
\sigma^2_{\text{max}} = \lambda_1
$$

## 4. 代码复现

```python
import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 1. 生成模拟数据
np.random.seed(42)
mean = [0, 0]
cov = [[3, 1], [1, 2]]  # 协方差矩阵
data = np.random.multivariate_normal(mean, cov, 200)

# 2. 计算协方差矩阵
C = np.cov(data.T)

# 3. 特征值分解
eigvals, eigvecs = np.linalg.eig(C)

# 4. 按特征值大小排序
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# 5. 主成分
max_pc = eigvecs[:, 0]      # 第一主成分
second_pc = eigvecs[:, 1]   # 第二主成分

# 6. 可视化
plt.figure(figsize=(6,6))
plt.scatter(data[:,0], data[:,1], alpha=0.3, label="样本数据")
origin = np.mean(data, axis=0)

# 绘制第一主成分
pc1_line = np.vstack((origin - max_pc*3, origin + max_pc*3))
plt.plot(pc1_line[:,0], pc1_line[:,1], 'r-', linewidth=2, label="第一主成分")

# 绘制第二主成分
pc2_line = np.vstack((origin - second_pc*3, origin + second_pc*3))
plt.plot(pc2_line[:,0], pc2_line[:,1], 'b--', linewidth=2, label="第二主成分")

plt.axis('equal')
plt.legend()
plt.title("PCA 第一与第二主成分可视化")
plt.show()

```