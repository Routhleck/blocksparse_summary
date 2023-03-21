---
title: void Compute
date: 2023-03-20T16:22:16Z
lastmod: 2023-03-20T16:22:38Z
---

# void Compute

```cpp
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
```
