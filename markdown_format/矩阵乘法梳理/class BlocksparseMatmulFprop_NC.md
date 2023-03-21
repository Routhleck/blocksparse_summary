---
title: class BlocksparseMatmulFprop_NC
date: 2023-03-21T14:10:25Z
lastmod: 2023-03-21T14:10:34Z
---

# class BlocksparseMatmulFprop_NC

‍

‍

‍

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulFprop_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulFprop_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "fprop", 32, 128) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        return this->Xprop_Kernel(A, B, C);
    }
};
```
