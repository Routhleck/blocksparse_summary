---
title: void Compute
date: 2023-03-20T16:27:30Z
lastmod: 2023-03-21T17:11:58Z
---

# void Compute

实现了具体的计算逻辑  
调用 CUDA 函数IdentityInitCK来生成权重矩阵并将其存储在输出张量 `w`​ 中。

```cpp
void Compute(OpKernelContext* ctx) override {

    TensorShape c_shape({ blocks_, bsize_, bsize_ });

    Tensor* w = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &w));

        float*   w_ptr = w->flat<float>().data();
    const int* lut_ptr = ctx->input(0).flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    IdentityInitCK(stream, w_ptr, lut_ptr, CB_, KB_, blocks_, bsize_, scale_);
  }
```

‍
