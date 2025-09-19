---
title: Algorithms_1 数组
date: 2025-09-17 15:00:00
updated: 2025-09-17 15:00:00
tags:
  - Algorithms
categories:
  - 数学
description: 时间复杂度
---


##  1. 数组(array)

数组是一个容器,用来存放一组相同类型的数据, 数组是存放在连续空间上的相同类型数据的集合

数组的属性: 1. 内存地址连续 2. 具有下标 3. 数组的元素不能删, 只能覆盖 4. 增删时,要移动其他元素的地址

### 一维数组

<img src="https://file1.kamacoder.com/i/algo/%E7%AE%97%E6%B3%95%E9%80%9A%E5%85%B3%E6%95%B0%E7%BB%84.png" alt="img" style="zoom: 50%;" />

```c++
#include <iostream>
// 声明一个包含 5 个整数的一维数组，代表 5 位学生的分数
int scores[5] = {98, 87, 92, 79, 85};

// 访问第 3 个学生的分数 (索引从 0 开始，所以是 scores[2])
int thirdStudentScore = scores[2]; // 结果是 92
```

### 二维数组

<img src="https://file1.kamacoder.com/i/algo/20201214111631844.png" alt="img" style="zoom: 25%;" />

```c++
// 声明一个 3 行 4 列的二维数组，代表一个班级的座位表
// 里面的每个数字可以代表学生的学号
int seatingChart[3][4] = {
    {101, 102, 103, 104},  // 第 0 行
    {201, 202, 203, 204},  // 第 1 行
    {301, 302, 303, 304}   // 第 2 行
};

// 访问第 1 行、第 2 列的那个学生 (索引从 0 开始)
int studentID = seatingChart[1][2]; // 结果是 203
```

##  2. 二分法 
[题目链接(704.) ](https://leetcode.cn/problems/binary-search/)[文章讲解   ](https://programmercarl.com/0704.二分查找.html)[视频讲解   ](https://www.bilibili.com/video/BV1fA4y1o715)


```c++
// 左闭右闭
class Solution{
public:
    int search(vector<int>& nums,int target){
        int left = 0;
        int right = nums.size() - 1;
        
        while (left <= right){
            int mid = (right - left) / 2 + left;
            
            if (nums[mid] > target){
                right = mid - 1;
            }else if (nums[mid] < target){
                left = mid + 1; 
            }else {
                return mid;
            }
        }
        return -1;
    }
};

```

```python
// 左闭右开
class Solution{
public:
    int search(vector<int>& nums,int target){
        int left = 0;
        int right = nums.size();
        
        while (left < right){
            int mid = (right - left) / 2 + left;
            
            if (nums[mid] > target){
                right = mid;
            }else if(nums[mid] < target){
                left = mid + 1;
            }else{
                return mid;
            }
        }
        return -1;
    }
};
```

## 3. 移除元素

[题目链接(27.)  ](https://leetcode.cn/problems/remove-element/) [文章讲解   ](https://programmercarl.com/0027.移除元素.html)[视频讲解    ](https://www.bilibili.com/video/BV12A4y1Z7LP) 

```python
// 快慢指针,也可以暴力解法
// 核心思想,覆盖
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int slow = 0;
        for (int fast = 0; fast < nums.size(); fast++){
            if (nums[fast] != val){
                nums[slow] = nums[fast];
                slow ++;
            }
        }
        return slow;
    }
};
```

<img src="https://file1.kamacoder.com/i/algo/27.%E7%A7%BB%E9%99%A4%E5%85%83%E7%B4%A0-%E5%8F%8C%E6%8C%87%E9%92%88%E6%B3%95.gif" alt="双指针" style="zoom: 80%;" />

## 4. 有序数组的平方

[题目链接   ](https://leetcode.cn/problems/squares-of-a-sorted-array/)[文章讲解  ](https://programmercarl.com/0977.%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E5%B9%B3%E6%96%B9.html)[视频讲解    ](https://www.bilibili.com/video/BV1QB4y1D7ep) 

```c++
// 思路 开方+快排, 时间复杂度为 O(Nlogn)
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        for (int &num : nums){
            num = num * num;
        }

    std::sort(nums.begin(), nums.end());

    return nums;
    }
};
```

```c++
//双指针, 有序, 递减缺一不可
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        //定义一个新的数组来装这个容器
        int n = nums.size();
        std::vector<int> result(n);

        //定义头尾指针
        int left = 0;
        int right = n - 1;

        // k 指针用于从后往前填充results的数组
        for (int k = n - 1; k >= 0; k--){
            // 比较两端元素的绝对值大小
            if (std::abs(nums[left]) > std::abs(nums[right])){
                result[k] = nums[left] * nums[left];
                left ++;
            }else{ 
                result[k] = nums[right] * nums[right];
                right --;
            }
        }
    return result;
    }
};
```

![](https://file1.kamacoder.com/i/algo/977.%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E5%B9%B3%E6%96%B9.gif)

##  总结

**基础概念先行**：夯实了数组的基础知识，明确了其**内存连续、通过下标访问、元素不可删除只能覆盖**等关键特性，为后续的算法学习打下了坚实的基础。

**经典算法剖析**：笔记深入探讨了三种基于数组的经典算法问题：

- **二分查找**：通过“左闭右闭” `[left, right]` 和“左闭右开” `[left, right)` 两种区间的代码实现，清晰地展示了二分法在处理边界条件时的细节与不同写法，这是掌握该算法的关键。
- **移除元素**：重点介绍了**快慢指针**技巧。通过让慢指针指向有效数组的末尾，快指针负责遍历查找，巧妙地实现了原地移除元素，将时间复杂度优化到 O(n)。
- **有序数组的平方**：本例是点睛之笔。文章对比了两种解法：一是直观的**“先平方，后排序”**（时间复杂度 O(n log n)），二是更优的**“头尾双指针”**法。后者充分利用了原数组的有序特性，从数组两端向中间比较并填充新数组，仅需一次遍历即可完成，将时间复杂度降至 O(n)，充分展现了算法优化的魅力。
- **题目推荐**:704, 35, 34, 69, 367, 27, 977, 26, 283, 844

###  感悟

- 先整体后部分, 要清楚的知道数据的流动性
- 基础知识要掌握, 思考这道题应该怎么做, 为什么这么做, 还有别的方法没有