---
title: Algorithms_2 数组2 
date: 2025-09-18 15:00:00
updated: 2025-09-18 15:00:00
tags:
  - Algorithms
categories:
  - 数学
description: 数组2
---
##  1. 滑动窗口(长度最小的子数组)

[题目链接	](https://leetcode.cn/problems/minimum-size-subarray-sum/)[文章讲解	](https://programmercarl.com/0209.%E9%95%BF%E5%BA%A6%E6%9C%80%E5%B0%8F%E7%9A%84%E5%AD%90%E6%95%B0%E7%BB%84.html)[视频讲解](https://www.bilibili.com/video/BV1tZ4y1q7XE)

最开始想到的解法,双层for循环暴力破解, 但时间复杂度为$O(N^2)$

```python
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int result = INT32_MAX; // 取int最大值
        int sum = 0; // 子序列数值之和
        int subLength = 0; // 子序列长度
        for (int i = 0; i < nums.size(); i++){
            sum = 0; // 每次循环sum = 0
            for (int j = i; j < nums.size(); j++){
                sum += nums[j];
                if (sum >= target){
                    subLength = j - i + 1;
                    result = result < subLength ? result : subLength;
                    break;
                }
            }
        }
        return result == INT32_MAX ? 0: result; 
        
    }
};
```

**滑动窗口** , 核心思想: 只关注最有潜力的部分，及时丢弃无效的部分

数据的流动性, 你要清楚的知道框的移动

用两个指针（`left` 和 `right`）来维护一个“窗口”，通过移动这两个指针来动态调整窗口的大小，以解决连续子数组或子串的问题。

<img src="https://file1.kamacoder.com/i/algo/209.%E9%95%BF%E5%BA%A6%E6%9C%80%E5%B0%8F%E7%9A%84%E5%AD%90%E6%95%B0%E7%BB%84.gif" style="zoom:50%;" />

```c++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int result = INT32_MAX; //结果
        int sum = 0; //子集合总和
        int subLength = 0; //子集合长度
        int left = 0; //左窗口
        for (int right = 0; right < nums.size(); right++){
            sum += nums[right]; //right窗口移动
            while (sum >= target){
                subLength = right - left + 1;
                result = result < subLength ? result : subLength;
                sum -= nums[left]; //left做窗口移动
                left++;
            }
        }
        return result == INT32_MAX ? 0 : result;
    }
};
```

##  2. **螺旋矩阵II(二维数组)**

[题目链接	](https://leetcode.cn/problems/spiral-matrix-ii/)[文章讲解	](https://programmercarl.com/0059.%E8%9E%BA%E6%97%8B%E7%9F%A9%E9%98%B5II.html)[视频讲解](https://www.bilibili.com/video/BV1SL4y1N7mV/)

循环不变量: 循环不变量是在循环执行过程中，**始终保持为真**的一个逻辑条件，它用于证明算法的正确性。

```python
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        
        vector<vector<int>> res(n, vector<int>(n, 0)); //创建一个二维数组
        int startx = 0, starty = 0; // 每循环一个圈的起始位置
        int loop = n / 2; //总共循环多少圈, n = 3, loop = 1.
        int mid = n / 2; //矩阵中间的位置, 矩阵中间的位置,用来给奇数中心赋值地址
        
        int count = 1; // 给新数组每个位置添加数值的
        int offset = 1; // 需要控制每一条边遍历的长度，每次循环右边界收缩一位.
        int i,j;

        while (loop--){
            i = startx;
            j = starty;

            // 左到右
            for (j; j < n - offset; j++){
                res[i][j] = count++;
            }
            // 右上到右下
            for (i; i < n -offset; i ++){
                res[i][j] = count++;
            }
            // 右到左
            for (j; j > starty; j--){
                res[i][j] = count++;
            }
            // 左下到左上
            for (i; i > startx; i-- ){
                res[i][j] = count++;
            }

            // 第二圈开始, 起始位置要加一
            startx++;
            starty++;
            offset += 1;
        }
        if (n % 2){
            res[mid][mid] = count;
        }
        return res;
    }
};
```

##  3. 前缀和

[文章	](https://www.programmercarl.com/kamacoder/0058.%E5%8C%BA%E9%97%B4%E5%92%8C.html#%E6%80%9D%E8%B7%AF)[题目](https://kamacoder.com/problempage.php?pid=1044)

**预先计算并存储一个数组（或序列）中每个位置之前所有元素的累加和，以便在后续需要查询任意子数组的和时，能够以极快的速度（O(1) 时间复杂度) 得到结果** 以空间换时间的方法

```c++
// 暴力解法
#include <iostream>
#include <vector>
using namespace std;
int main() {
    int n, a, b;
    cin >> n;
    vector<int> vec(n);
    for (int i = 0; i < n; i++) cin >> vec[i];
    while (cin >> a >> b) {
        int sum = 0;
        // 累加区间 a 到 b 的和
        for (int i = a; i <= b; i++) sum += vec[i];
        cout << sum << endl;
    }
} 
```

```python
// 前缀和
#include <iostream>
#include <vector>
using namespace std;
int main() {
    int n, a, b;
    cin >> n;
    vector<int> vec(n);
    vector<int> p(n);
    int presum = 0;
    for (int i = 0; i < n; i++) {
        cin >> vec[i];
        presum += vec[i];
        p[i] = presum;
    }

    while (cin >> a >> b) {
        int sum;
        if (a == 0) sum = p[b];
        else sum = p[b] - p[a - 1];
        cout << sum << endl;
    }
}
```

