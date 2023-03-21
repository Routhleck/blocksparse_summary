---
title: class BlocksparseMatmulBprop_NC
date: 2023-03-21T14:10:43Z
lastmod: 2023-03-21T14:10:52Z
---

# class BlocksparseMatmulBprop_NC

‍

‍

‍

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
