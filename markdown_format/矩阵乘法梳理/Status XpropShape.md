---
title: Status XpropShape
date: 2023-03-20T15:13:14Z
lastmod: 2023-03-20T15:13:37Z
---

# Status XpropShape

```cpp
Status XpropShape(InferenceContext* ctx)
{
    int    K; TF_RETURN_IF_ERROR(ctx->GetAttr(   "K",    &K));
    int axis; TF_RETURN_IF_ERROR(ctx->GetAttr("axis", &axis));

    // C ==> K
    ShapeHandle x = ctx->input(0);
    int rank = ctx->Rank(x);
    //printf("XpropShape: %d\n", rank);
    if (rank > 0)
    {
        std::vector<DimensionHandle> shape;
        shape.reserve(rank);
        for (int i = 0; i < rank; i++)
            shape.push_back(i == axis ? ctx->MakeDim(K) : ctx->Dim(x, i));

        ctx->set_output(0, ctx->MakeShape(shape));
    }
    else
        ctx->set_output(0, ctx->UnknownShape());
    ctx->set_output(1, ctx->UnknownShape());
    return Status::OK();
}
```
