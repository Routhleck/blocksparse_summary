---
title: explicit BlocksparseMatmulIdentityInitOp
date: 2023-03-20T16:27:15Z
lastmod: 2023-03-21T17:09:21Z
---

# explicit BlocksparseMatmulIdentityInitOp

从OpKernelConstruction中获取上下文对象ctx，并从输入参数中获取操作的一些属性

```cpp
explicit BlocksparseMatmulIdentityInitOp(OpKernelConstruction* ctx) : OpKernel(ctx) {

    OP_REQUIRES_OK(ctx, ctx->GetAttr("CB",     &CB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("KB",     &KB_    ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks", &blocks_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",  &bsize_ ));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("scale",  &scale_ ));
  }
```
