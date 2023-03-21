---
title: class BlocksparseMatmulIdentityInitOp
date: 2023-03-20T15:15:52Z
lastmod: 2023-03-20T16:35:12Z
---

# class BlocksparseMatmulIdentityInitOp

# 成员方法

|成员方法|返回类型|说明|
| ---------------------------------| ----------| -------------------------------------------------------------------------------------------------------|
|BlocksparseMatmulIdentityInitOp|构造函数|构造函数，初始化`CB_`​,`KB_`​,`blocks_`​,`bsize_`​,`scale_`​属性|
|Compute|void|实现矩阵乘法并初始化输出矩阵的对角线元素为1。该方法使用输入矩阵`lut_ptr`​以及属性`CB_`​,`KB_`​,`blocks_`​,`bsize_`​,`scale_`​计算输出矩阵`w`​|

# 成员变量

|成员变量|类型|说明|
| ----------| ------| ----------------------------|
|​`blocks_`​|​`int`​|矩阵分块的数量|
|​`bsize_`​|​`int`​|每个分块矩阵的大小|
|​`CB_`​|​`int`​|输入矩阵的列数|
|​`KB_`​|​`int`​|输入矩阵的行数|
|​`scale_`​|​`float`​|初始化对角线元素的比例因子|

# 具体代码

```cpp
class BlocksparseMatmulIdentityInitOp : public OpKernel {
 public:
  explicit BlocksparseMatmulIdentityInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("CB",     &CB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("KB",     &KB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks", &blocks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale",  &scale_ ));
  }

  void Compute(OpKernelContext* ctx) override {

    TensorShape c_shape({ blocks_, bsize_, bsize_ });

    Tensor* w = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, c_shape, &w));

        float*   w_ptr = w->flat<float>().data();
    const int* lut_ptr = ctx->input(0).flat<int32>().data();

    CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

    IdentityInitCK(stream, w_ptr, lut_ptr, CB_, KB_, blocks_, bsize_, scale_);
  }
 private:

  int blocks_, bsize_, CB_, KB_;
  float scale_;
};
```
