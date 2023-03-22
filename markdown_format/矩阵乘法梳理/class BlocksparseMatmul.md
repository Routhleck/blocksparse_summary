---
title: class BlocksparseMatmul
date: 2023-03-21T14:08:26Z
lastmod: 2023-03-22T14:55:26Z
---

# class BlocksparseMatmul

基类，由子类BlocksparseMatmulFprop_CN, BlocksparseMatmulBprop_CN, BlocksparseMatmul_NC, 以及另一个BlocksparseMatmul_NC继承(此类也是用作基类，被其他类继承)

# 成员方法

|成员方法|返回类型|说明|
| --------------------| ----------| ----------------------------|
|BlocksparseMatmul|构造函数|构造函数，初始化类成员变量|
|~BlocksparseMatmul|析构函数|析构函数|
|Compute|status|矩阵乘法计算|

# 成员变量

|成员变量|类型|说明|
| ----------| --------------| --------------|
|params|bsmm_params*|参数结构体|
|major_|int|CUDA主版本号|

# 具体代码

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmul
{
public:
    BlocksparseMatmul(bsmm_params* params) : params_(params), major_(0) {}
    virtual ~BlocksparseMatmul() {}

    virtual Status Compute(const TA* A, const TB* B, TC* C) =0;

    bsmm_params* params_;
    int major_;
};
```
