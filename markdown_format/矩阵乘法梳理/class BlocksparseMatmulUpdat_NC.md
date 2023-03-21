---
title: class BlocksparseMatmulUpdat_NC
date: 2023-03-21T14:10:59Z
lastmod: 2023-03-21T14:11:10Z
---

# class BlocksparseMatmulUpdat_NC

‍

‍

‍

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmulUpdat_NC : public BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmulUpdat_NC(bsmm_params* params) :
        BlocksparseMatmul_NC<VTYPE3(TA,TB,TC)>(params, "updat", 8, 32) {}

    virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        struct plist8<TA>* pA = (struct plist8<TA>*)A;
        struct plist8<TB>* pB = (struct plist8<TB>*)B;
        bsmm_params* params = this->params_;
        int pcount = params->pcount * 8;

        //printf("%p %p %p %p %d %d\n", pA->a[0], pB->a[0], L, C, N, params);

        GetKernel(this->kernel_name_, &this->kernel_);
        //printf("%s %p\n", kernel_name_.c_str(), kernel_);

        void *args[] = { pA, pB, &params->Lut, &C, &params->alpha, &params->beta, &params->C, &params->K, &params->N, &pcount };

        CUresult res = cuLaunchKernel(this->kernel_, params->blocks, 1, 1, this->threads_, 1, 1, params->shared, params->stream, args, NULL);
        if (res != CUDA_SUCCESS)
        {
            const char* errstr;
            cuGetErrorString(res, &errstr);
            return errors::Internal(errstr);
        }
        return Status::OK();
    }
};
```
