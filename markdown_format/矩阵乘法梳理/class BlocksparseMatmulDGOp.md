---
title: class BlocksparseMatmulDGOp
date: 2023-03-20T15:14:02Z
lastmod: 2023-03-22T14:56:58Z
---

# class BlocksparseMatmulDGOp

实现反向传播操作，计算某个神经网络层的权重矩阵和梯度的矩阵乘法，以及相应的梯度计算

# 成员方法

|成员方法|返回类型|说明|
| -----------------------| ----------| --------------|
|BlocksparseMatmulDGOp|构造函数|初始化`BlocksparseMatmulDGOp`​对象|
|Compute|void|计算梯度值|

# 成员变量

|成员变量|变量类型|说明|
| ---------------------------| ---------------| ------------------------------|
|​dw_out​​|Tensor*|分配输出张量，梯度值的张量|
|​​dg​​|Tensor*|分配输出张量，门控梯度的张量|
|​dw​​|const Tensor&|输入张量，权重梯度的张量|
|​w​​|const Tensor&|输入张量，权重的张量|
|​g​​|const Tensor&|输入张量，门控的张量|
|​stream​​|CUstream|计算所在的 CUDA 流|
|​BlocksparseGateGrad​​|void|计算门控梯度和梯度值|

# 具体代码

```cpp
template <typename T, typename V>
class BlocksparseMatmulDGOp : public OpKernel {
 public:
  explicit BlocksparseMatmulDGOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }

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
};
```

‍
