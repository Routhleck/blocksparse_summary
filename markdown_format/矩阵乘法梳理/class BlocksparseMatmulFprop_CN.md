---
title: class BlocksparseMatmulFprop_CN
date: 2023-03-21T14:09:01Z
lastmod: 2023-03-22T14:58:14Z
---

# class BlocksparseMatmulFprop_CN

继承自父类BlocksparseMatmul，并实现前向传播操作

基本与父类一致，仅仅实现了Compute函数

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulFprop_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulFprop_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        if (this->major_ == 0)
            GetCountSMsVersion(&this->major_, NULL);

        cudaError_t res;
        if (this->major_ >= 7 && std::is_same<TA, ehalf>::value)
            res = hgemm_blocksparse_xn_sdd(A, B, C, this->params_, 1);
        else
            if (this->params_->Gate == NULL)
                res = BsmmXprop_CN<true,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
            else
                res = BsmmGatedXprop_CN<true,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};
```
