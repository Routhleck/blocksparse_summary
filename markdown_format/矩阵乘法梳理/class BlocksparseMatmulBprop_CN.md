---
title: class BlocksparseMatmulBprop_CN
date: 2023-03-21T14:09:14Z
lastmod: 2023-03-21T14:09:29Z
---

# class BlocksparseMatmulBprop_CN

‍

‍

‍

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulBprop_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulBprop_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        if (this->major_ == 0)
            GetCountSMsVersion(&this->major_, NULL);

        cudaError_t res;
        if (this->major_ >= 7 && std::is_same<TA, ehalf>::value)
            res = hgemm_blocksparse_xn_sdd(A, B, C, this->params_, 0);
        else
            if (this->params_->Gate == NULL)
                res = BsmmXprop_CN<false,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
            else
                res = BsmmGatedXprop_CN<false,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};
```
