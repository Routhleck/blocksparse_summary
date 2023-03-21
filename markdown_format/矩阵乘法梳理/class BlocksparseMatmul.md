---
title: class BlocksparseMatmul
date: 2023-03-21T14:08:26Z
lastmod: 2023-03-21T14:08:50Z
---

# class BlocksparseMatmul

‍

‍

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
