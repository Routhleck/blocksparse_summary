---
title: class BlocksparseMatmulBprop_NC
date: 2023-03-21T14:10:43Z
lastmod: 2023-03-22T15:22:01Z
---

# class BlocksparseMatmulBprop_NC

继承自父类BlocksparseMatmul_NC，并实现向前传播操作

与父类基本一致，不同之处在于它的构造函数调用了父类的构造函数，并传递了参数 `"bprop"`​、`32`​ 和 `128`​，这些参数指定了稀疏矩阵乘法中使用的 CUDA kernel 的操作类型、深度和线程数。

同时重写了Compute函数

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulBprop_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulBprop_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "bprop", 32, 128) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        return this->Xprop_Kernel(A, B, C);
    }
};
```
