---
title: Status UpdatShape
date: 2023-03-20T15:13:44Z
lastmod: 2023-03-20T15:14:00Z
---

# Status UpdatShape

```cpp
Status UpdatShape(InferenceContext* ctx)
{
    //printf("UpdatShape: %d\n", ctx->Rank(ctx->input(0)));

    int blocks, bsize;
    TF_RETURN_IF_ERROR(ctx->GetAttr("blocks", &blocks));
    TF_RETURN_IF_ERROR(ctx->GetAttr("bsize",  &bsize));

    // (blocks, block_size, block_size)
    DimensionHandle bsize_dim = ctx->MakeDim(bsize);
    ctx->set_output(0, ctx->MakeShape({ ctx->MakeDim(blocks), bsize_dim, bsize_dim }));
    return Status::OK();
}
```

‍
