---
title: Status Compute
date: 2023-03-22T15:16:06Z
lastmod: 2023-03-22T15:16:41Z
---

# Status Compute

调用了父类中的 ​Xprop_Kernel 函数，用于执行前向传播操作的稀疏矩阵乘法。

```cpp
virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        return this->Xprop_Kernel(A, B, C);
    }
```
