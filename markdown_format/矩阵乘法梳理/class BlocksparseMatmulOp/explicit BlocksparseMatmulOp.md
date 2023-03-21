---
title: explicit BlocksparseMatmulOp
date: 2023-03-20T15:59:09Z
lastmod: 2023-03-21T14:21:31Z
---

# explicit BlocksparseMatmulOp

class BlocksparseMatmulOp的构造函数，初始化各种参数

```cpp
explicit BlocksparseMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), major_(0), repeat_(1), flops_(0.0f)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("segments", &params_.segments));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("locks",    &params_.locks   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &params_.blocks  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",    &params_.bsize  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("C",        &params_.C       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("K",        &params_.K       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shared",   &params_.shared  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",    &params_.alpha   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta",     &params_.beta    ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gated_dw", &gated_dw_       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_));
        params_.pcount = 1;
        params_.blk_A  = 0;

        is_gpu_ = ctx->device_type() == DEVICE_GPU;

        //OP_REQUIRES(ctx, axis_ == 0, errors::InvalidArgument("Only feature axis=0 currently supported."));

        // TODO: pack larger values of K in gridZ
        OP_REQUIRES(ctx, params_.K < params_.bsize*65536, errors::InvalidArgument("K < bsize*65536"));
        OP_REQUIRES(ctx, params_.C < params_.bsize*65536, errors::InvalidArgument("C < bsize*65536"));

        if (bench_)
        {
            repeat_ = bench_;
            flops_  = (float)(params_.blocks * params_.bsize*params_.bsize);

            const char* op = OP == FPROP_OP ? "FPROP" : OP == BPROP_OP ? "BPROP" : "UPDAT";
            sprintf(bench_string_, "%s %02d-%d C:%05d K:%05d blks:%d", op, params_.bsize, axis_, params_.C, params_.K, params_.blocks);
        }
    }
```
