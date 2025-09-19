---
title: Algorithms_0 c++数据结构基础
date: 2025-09-15 15:00:00
updated: 2025-09-15 15:00:00
tags:
  - Algorithms
categories:
  - 数学
description: 时间复杂度
---
# 线性表

线性表是一个抽象的概念，它指的是一种数据结构，其中的数据元素是像一条线一样排列起来的，每个元素都有一个唯一的前驱和唯一的后继（除了第一个和最后一个元素）。

**数组(顺序表) 和 链表 都是线性表的 具体实现方式**

- 数组用连续的内存空间来存储数据, 是一种**顺序存储**的线性表
- 链表用离散的内存空间来存储数据,并通过指针将他们链接起来, 是一种**链式存储**的线性表

# 指针

**指针是用来存放内存地址的变量**

```c++
#include <iostream>

int main(){
    int a = 5;
    int *p = &a;
    printf("a的地址为: %p, a的值为: %d\n", &a, a);
    printf("p的地址为: %p, p的值为: %p\n", &p, p);
    return 0;
}

// 输出为
// a的地址为: 0x7ffef84c0e4c, a的值为: 5
// p的地址为: 0x7ffef84c0e50, p的值为: 0x7ffef84c0e4c
```

**间接引用操作符 **

间接引用操作符 * **返回指针变量的指向地址的值**, 通常把这个操作叫做"解引用指针", 只要前面没有带数据类型, 则为 解引用指针

```c++
#include <iostream>

int main(){
    int a = 5;
    int *p = &a;
    printf("%d\n", *p);
    *p = 100;
    printf("%d\n", a);
    return 0;
}
// 输出为: 5 100
```

**对指针做算数运算, 实际上加的是这个整数和指针数据类型对应的字节数的乘积.**

# 结构体

**结构题是一个或多个变量的集合, 这些变量可以是不同的类型.**

```c++
// 结构体语法

struct 结构体名{
    数据类型 变量名1;
    数据类型 变量名2;
}

// 初始化调用
struct 结构体名 变量名;
```

**结构体与指针配合使用**

```c++
#include <iostream>

struct point{ // 结构体
    int x;
    int y;
};

int main(){
    struct point  p; //创建 p 结构体
    p.x = 5;
    p.y = 10;

    struct point *pp; // 创建 指针结构体
    pp = &p;

    pp->x = 10;
    pp->y = 5;

    printf("x = %d, y = %d\n", p.x, p.y);
    printf("x = %d, y = %d\n", pp->x, pp->y);
    return 0;
}
```

# 类型定义

```c++
// 使用场景, 快速改变数据类型
typedef 数据类型 别名
```

**结构体与类型定义一起使用**

```c++
typedef struct{
    数据类型 变量名1;
    数据类型 变量名2;
}别名;
```

```python
#include <iostream>

typedef struct{ // 结构体
    int x;
    int y;
}point;

int main(){
    point p;
    p.x = 5;
    p.y =10;

    point *pp;
    pp = &p;
    pp->x = 10;
    pp->y = 5;

    printf("x = %d, y = %d\n", p.x, p.y);
    printf("x = %d, y = %d\n", pp->x, pp->y);

    return 0;
}
```

# 动态内存的分配

静态/全局内存: 静态声明变量和全局变量使用这部分内存,这些变量在程序运行时分配,直到程序结束消失

自动内存(栈内存): 函数内部声明的变量使用这部分内存, 在函数调用时才创建

动态内存(堆内存): 根据需求编写代码动态分配内存,可以编写代码释放, 内存重点内容知道释放才消失.

栈内存是系统自动管理的**小块**、**临时**内存，而堆内存是你手动管理的**大块**、**长期**内存。

**开辟内存**

1. 使用malloc函数分配内存

    ```c++
    void* malloc(size) 
    //成功分配, 失败返回空指针(NULL)
    ```

    

2. 使用分配的内存

3. 使用free函数释放内存  

```c++
#include <iostream>
// int整形
int mian(){
    int *p;
    p = (int*)malloc(sizeof(int));
    *p = 15;
    printf("%d\n", *p);
    free(p);
    return 0;
}

//char
#include <iostream>
#include <cstring>

int main(){
    char *s;
    s = (char*)malloc(10);
    strcpy(s, "hello");
    printf("%s\n",s);
    return 0;
}
```

线性表作为一种抽象概念，通过**数组（顺序存储）和链表（链式存储）两种方式具体实现**；而**指针**作为连接链表节点的关键，是存储内存地址的变量，能通过解引用操作符`*`访问其指向的值；**结构体**允许组合不同类型的数据，并常与指针配合使用，同时**`typedef`**能为这些复杂类型创建别名，简化代码；最后，**动态内存分配**（如`malloc`和`free`）则提供了在程序运行时按需管理内存的能力，是构建复杂数据结构不可或缺的工具。