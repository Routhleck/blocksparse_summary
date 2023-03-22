---
title: void Compute
date: 2023-03-20T16:16:34Z
lastmod: 2023-03-21T16:39:25Z
---

# void Compute

主要处理各种参数然后调用BlocksparseGateGrad来进行计算，主要是求出`dw_out`​和`dg`​

​`dw_out`​的计算结果是**反向传播时的梯度值**，它的形状与输入的`dw`​一致，代表着原始输入的梯度信息。

​`dg`​的计算结果是BlocksparseMatmul的**Gate（门控）的梯度值**，它的形状与输入的`g`​一致，代表着原始输入Gate的梯度信息。Gate的作用是控制前向传播中的哪些数据需要被过滤掉，因此它的梯度信息非常重要，可以帮助我们更好地训练模型。

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
