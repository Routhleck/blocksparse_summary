---
title: class BlocksparseMatmul_NC
date: 2023-03-21T14:09:54Z
lastmod: 2023-03-21T14:10:11Z
---

# class BlocksparseMatmul_NC

‍

‍

```cpp
template <CTYPE3(TA,TB,TC)>
class BlocksparseMatmul_NC : public BlocksparseMatmul<VTYPE3(TA,TB,TC)>
{
public:
    BlocksparseMatmul_NC(bsmm_params* params, const char* op, int depth, int threads) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params), threads_(threads)
    {
        const char* dtypeA = std::is_same<TA, ehalf>::value ? "A10" : std::is_same<TA, bhalf>::value ? "A7": "A32";
        const char* dtypeB = std::is_same<TB, ehalf>::value ? "B10" : std::is_same<TB, bhalf>::value ? "B7": "B32";
        const char* dtypeC = std::is_same<TC, ehalf>::value ? "C10" : std::is_same<TC, bhalf>::value ? "C7": "C32";

        // int depth;
        // const char* op;
        // if      (mode_  == 0) { op = "fprop"; depth = 32; threads_ = 128; }
        // else if (mode_  == 1) { op = "bprop"; depth = 32; threads_ = 128; }
        // else                  { op = "updat"; depth =  8; threads_ =  32; }

        char kernel_name[48];
        sprintf(kernel_name, "gemm_blocksparse_32x32x%d_%s_%s_%s_%s", depth, op, dtypeA, dtypeB, dtypeC);
        kernel_name_ = kernel_name;
        kernel_ = 0;
    }
    Status Xprop_Kernel(const TA* A, const TB* B, TC* C)
    {
        GetKernel(kernel_name_, &kernel_);
        //printf("%s %p\n", kernel_name_.c_str(), kernel_);

        bsmm_params* params = this->params_;

        int gridX = (params->N >> 5) + ((params->N & 31) != 0);
        int gridY = (params->K >> 5);

        void *args[] = { &params->Lut, &C, &A, &B, &params->alpha, &params->beta, &params->C, &params->K, &params->N };

        CUresult res = cuLaunchKernel(kernel_, gridX, gridY, 1, threads_, 1, 1, params->shared, params->stream, args, NULL);
        if (res != CUDA_SUCCESS)
        {
            const char* errstr;
            cuGetErrorString(res, &errstr);
            return errors::Internal(errstr);
        }
        return Status::OK();
    }
    virtual Status Compute(const TA* A, const TB* B, TC* C) =0;

    int threads_, gridX_, gridY_;
    std::string kernel_name_;
    CUfunction kernel_;
};
```
