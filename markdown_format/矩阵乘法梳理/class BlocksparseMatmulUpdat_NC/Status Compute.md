---
title: Status Compute
date: 2023-03-22T15:22:56Z
lastmod: 2023-03-22T15:24:09Z
---

# Status Compute

​`Compute`​ 函数接收三个参数 `A`​、`B`​ 和 `C`​，它们分别表示反向传播操作中的梯度、权重和误差。在这个函数中，`A`​ 和 `B`​ 的类型为 `TA`​ 和 `TB`​，但是它们实际上被转换为了 `plist8<TA>`​ 和 `plist8<TB>`​，因为在反向传播操作中，矩阵 A 和 B 是按列划分成了 8 个部分，每个部分包含 32 行。因此，在执行反向传播操作时，需要使用这些列向量。

接下来，`Compute`​ 函数获取了当前对象中保存的 CUDA kernel，并将输入参数和 CUDA kernel 的参数打包在一起。然后，它使用 CUDA runtime API 中的 `cuLaunchKernel`​ 函数来启动 CUDA kernel。

```cpp
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
```
