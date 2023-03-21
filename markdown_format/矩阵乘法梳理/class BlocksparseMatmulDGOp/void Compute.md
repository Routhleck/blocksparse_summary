---
title: void Compute
date: 2023-03-20T16:16:34Z
lastmod: 2023-03-20T16:17:02Z
---

# void Compute

```cpp
void Compute(OpKernelContext* ctx) override
  {
    const Tensor& dw = ctx->input(0);
    const Tensor&  w = ctx->input(1);
    const Tensor&  g = ctx->input(2);

    uint blocks = dw.dim_size(0);
    uint bsize  = dw.dim_size(1);

    Tensor *dw_out;
    Tensor *dg;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, dw.shape(), &dw_out));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1,  g.shape(), &dg));

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    BlocksparseGateGrad<V>(stream,
      (V*)dw_out->flat<T>().data(),
      dg->flat<float>().data(),
      (const V*)dw.flat<T>().data(),
      (const V*) w.flat<T>().data(),
      g.flat<float>().data(),
      blocks, bsize
    );
  }
```
