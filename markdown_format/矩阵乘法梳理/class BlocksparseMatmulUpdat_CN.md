---
title: class BlocksparseMatmulUpdat_CN
date: 2023-03-21T14:09:41Z
lastmod: 2023-03-22T15:08:48Z
---

# class BlocksparseMatmulUpdat_CN

继承自父类BlocksparseMatmul，并实现更新参数操作

基本与父类一致，仅仅实现了Compute函数

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulUpdat_CN : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulUpdat_CN(bsmm_params* params) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        if (this->major_ == 0)
            GetCountSMsVersion(&this->major_, NULL);

        cudaError_t res;
        if (this->major_ >= 7 && std::is_same<TA, ehalf>::value)
            res = hgemm_blocksparse_nt_dds(A, B, C, this->params_);
        else
            if (this->params_->Gate == NULL)
                res = BsmmUpdat_CN<VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
            else
                res = BsmmGatedUpdat_CN<VTYPE3(TA,TB,TC)>(A, B, C, this->params_);

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
};
```
