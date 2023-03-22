---
title: class BlocksparseMatmul_NC
date: 2023-03-21T14:09:54Z
lastmod: 2023-03-22T15:18:59Z
---

# class BlocksparseMatmul_NC

​`BlocksparseMatmul_NC`​ 是在 `BlocksparseMatmul`​ 的基础上实现了针对不同转置矩阵的稀疏矩阵乘法，**将根据不同的输入参数然后直接调用核函数，与之前的类有所不同，直接指定`kernel_name`**

# 成员方法

|成员方法|返回类型|说明|
| ----------------------| ----------| ----------------------------|
|BlocksparseMatmul_NC|构造函数|构造函数，初始化类成员变量|
|Xprop_Kernel|status|执行Kernel|
|Compute|status|矩阵乘法计算|

# 成员变量

|成员变量|类型|说明|
| --------------| --------------| -----------------------|
|threads_|int|CUDA thread 数量|
|gridX_|int|CUDA kernel 的 X 维度|
|gridY_|int|CUDA kernel 的 Y 维度|
|kernel_name_|std::string|CUDA kernel 的名称|
|kernel_|CUfunction|CUDA kernel 的句柄|
|params_|bsmm_params*|稀疏矩阵乘法的参数|
|major_|int|CUDA主版本号|

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
