---
title: Status Xprop_Kernel
date: 2023-03-22T14:38:21Z
lastmod: 2023-03-22T14:38:58Z
---

# Status Xprop_Kernel

执行Kernel

首先根据成员变量 `kernel_name_`​ 和 CUDA context，获取对应的 CUDA kernel，并计算出 kernel 执行时需要的参数。然后使用 CUDA API `cuLaunchKernel`​ 函数执行 kernel。

```cpp
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
```
