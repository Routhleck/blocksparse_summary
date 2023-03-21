---
title: explicit BlocksparseReducedDWOp
date: 2023-03-20T16:22:07Z
lastmod: 2023-03-20T16:22:25Z
---

# explicit BlocksparseReducedDWOp

```cpp
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
```
