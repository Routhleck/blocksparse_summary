---
title: class BlocksparseReducedDWOp
date: 2023-03-20T15:34:44Z
lastmod: 2023-03-20T16:34:42Z
---

# class BlocksparseReducedDWOp

# 成员方法

|成员方法|返回类型|说明|
| ------------------------| ----------| ------------------------------------------|
|BlocksparseReducedDWOp|构造函数|构造函数，初始化类成员变量|
|Compute<br />|void|进行前向计算，输出结果到 OpKernelContext|

# 成员变量

|成员变量|类型|说明|
| ----------| ------| --------------------------------------|
|​`bshift`​|​`int`​|用于指定块的大小，根据`bsize`​属性计算得出|
|​`norm`​|​`int`​|指定是否进行归一化|
|​`axis`​|​`int`​|指定计算的轴，只支持 0 或 1|
|​`major_version`​|​`int`​|指定 Tensorcore GPU 的版本|
|​`blk_A`​|​`uint`​|用于 cuBLAS 中的 GEMM 计算|
|​`blk_B`​|​`uint`​|用于 cuBLAS 中的 GEMM 计算|
|​`blk_a`​|​`uint`​|用于 cuBLAS 中的 GEMM 计算|
|​`blk_b`​|​`uint`​|用于 cuBLAS 中的 GEMM 计算|

# 具体代码

```cpp
class BlocksparseReducedDWOp : public OpKernel
{
public:
    explicit BlocksparseReducedDWOp(OpKernelConstruction* ctx) : OpKernel(ctx), major_version(0)
    {
        int bsize;
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize", &bsize));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("norm",  &norm ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",  &axis ));
        OP_REQUIRES(ctx, axis == 0 || axis == 1, errors::InvalidArgument("invalid feature axis, only 0,1 supported."));
        if (axis == 0)
            OP_REQUIRES(ctx, bsize == 8 || bsize == 16 || bsize == 32, errors::InvalidArgument("Only feature axis=0 supports blocksizes: 8,16,32"));
        else
            OP_REQUIRES(ctx, bsize == 32 || bsize == 64, errors::InvalidArgument("Only feature axis=0 supports blocksizes: 32,64"));

        bshift = bsize == 8 ? 3 : bsize == 16 ? 4 : bsize == 32 ? 5 : 6;
    }
    void Compute(OpKernelContext* ctx) override
    {
        OpInputList x, y;
        ctx->input_list( "x", &x);
        ctx->input_list("dy", &y);
        uint params = x.size();
        float scale = ctx->input(params*2).scalar<float>()();
        OP_REQUIRES(ctx, params <= 8, errors::InvalidArgument("No more than 8 inputs allowed."));

        uint C  = x[0].dim_size(axis);
        uint K  = y[0].dim_size(axis);
        uint bC = C >> bshift;
        uint bK = K >> bshift;
        uint N  = x[0].dim_size(1-axis);
        TensorShape shapeX, shapeY;
        if (axis == 0)
        {
            shapeX.AddDim(bC);
            shapeY.AddDim(bK);
        }
        shapeX.AddDim(params);
        shapeY.AddDim(params);
        shapeX.AddDim(N);
        shapeY.AddDim(N);
        if (axis == 1)
        {
            shapeX.AddDim(bC);
            shapeY.AddDim(bK);
        }

        if (major_version == 0)
        {
            GetCountSMsVersion(&major_version, NULL);
            OP_REQUIRES(ctx, major_version >= 7, errors::InvalidArgument("Tensorcore GPU required"));

            OP_REQUIRES(ctx, (bC & 1) == 0 && (bK & 1) == 0, errors::InvalidArgument("Block reduced feature dim must be multiple of 2."));

            ClosestDivisorTo4(axis == 0 ? CEIL_DIV(bC, 32) : CEIL_DIV(bC, 64), true, &blk_a, &blk_A);
            ClosestDivisorTo4(axis == 0 ? CEIL_DIV(bK, 32) : CEIL_DIV(bK, 64),false, &blk_b, &blk_B);
        }

        struct Plist<ehalf,8> X, Y;
        for (int i = 0; i < params; ++i)
        {
            X.a[i] = (const ehalf*)x[i].flat<EHALF>().data();
            Y.a[i] = (const ehalf*)y[i].flat<EHALF>().data();
        }

        float* DW;
        uint accumulate;
        if (ctx->num_inputs() > params*2 + 1)
        {
            // accumulate to DW in place
            accumulate = 1;
            const Tensor& dw = ctx->input(params*2 + 1);
            ctx->set_output(0, dw);
            DW = (float*)dw.flat<float>().data();
        }
        else
        {
            accumulate = 0;
            Tensor *dw;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({ bC, bK }), &dw));
            DW = dw->flat<float>().data();
        }
        Tensor *redX, *redY;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, shapeX, &redX));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, shapeY, &redY));
        ehalf* RedX = (ehalf*)redX->flat<EHALF>().data();
        ehalf* RedY = (ehalf*)redY->flat<EHALF>().data();

        CUstream stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        if (scale != 0.0f)
        {
            if (axis == 0)
            {
                BlocksparseFeatureReduceCN(stream, RedX, &X, params, C, N, bshift, norm);
                BlocksparseFeatureReduceCN(stream, RedY, &Y, params, K, N, bshift, norm);
            }
            else
            {
                BlocksparseFeatureReduceNC(stream, RedX, &X, params, C, N, bshift, norm);
                BlocksparseFeatureReduceNC(stream, RedY, &Y, params, K, N, bshift, norm);
            }
        }
        if (axis == 0)
            hGemmNT(stream, RedX, RedY, DW, bC, bK, N*params, blk_A, blk_B, blk_a, blk_b, accumulate, scale);
        else
            hGemmTN(stream, RedX, RedY, DW, bC, bK, N*params, blk_A, blk_B, blk_a, blk_b, accumulate, scale);
    }
    int  bshift, norm, axis, major_version;
    uint blk_A, blk_B, blk_a, blk_b;
};
```

‍
