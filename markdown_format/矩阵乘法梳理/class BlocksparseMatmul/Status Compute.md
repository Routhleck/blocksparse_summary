---
title: Status Compute
date: 2023-03-22T11:36:16Z
lastmod: 2023-03-22T14:58:34Z
---

# Status Compute

虚函数

# 输入参数

|参数|类型|说明|
| -------------------| -----------------| -------------------------|
|​​const TA* A​|​const​指针|稀疏矩阵 A 中的元素数组|
|​​const TB* B​|​​const​指针|稀疏矩阵 B 中的元素数组|

# 输出参数

|参数|类型|说明|
| -------| ------| -------------------------|
|TC* C|指针|稀疏矩阵 C 中的元素数组|

矩阵乘法`A`​x`B`​=`C`​

```cpp
virtual Status Compute(const TA* A, const TB* B, TC* C) =0;
```

‍
