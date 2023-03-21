---
title: Status ReducedDWShape
date: 2023-03-20T15:16:21Z
lastmod: 2023-03-20T15:16:32Z
---

# Status ReducedDWShape

```cpp
Status ReducedDWShape(InferenceContext* ctx)
{
    int params, bsize, axis;
    TF_RETURN_IF_ERROR(ctx->GetAttr("n_params", &params));
    TF_RETURN_IF_ERROR(ctx->GetAttr("bsize",    &bsize));
    TF_RETURN_IF_ERROR(ctx->GetAttr("axis",     &axis));
    int bshift = bsize == 8 ? 3 : bsize == 16 ? 4 : bsize == 32 ? 5 : 6;

    ShapeHandle x = ctx->input(0);
    ShapeHandle y = ctx->input(params);
    int rank = ctx->Rank(x);
    if (rank > 1)
    {
        DimensionHandle C = ctx->MakeDim(ctx->Value(ctx->Dim(x, axis)) >> bshift);
        DimensionHandle K = ctx->MakeDim(ctx->Value(ctx->Dim(y, axis)) >> bshift);
        DimensionHandle P = ctx->MakeDim(params);

        std::vector<DimensionHandle> x_red, y_red;
        x_red.reserve(rank + 1);
        y_red.reserve(rank + 1);
        if (axis == 0)
        {
            x_red.push_back(C);
            y_red.push_back(K);
        }
        x_red.push_back(P);
        y_red.push_back(P);
        x_red.push_back(ctx->Dim(x, 1-axis));
        y_red.push_back(ctx->Dim(y, 1-axis));
        if (axis == 1)
        {
            x_red.push_back(C);
            y_red.push_back(K);
        }

        ctx->set_output(0, ctx->MakeShape({ C, K }));
        ctx->set_output(1, ctx->MakeShape(x_red));
        ctx->set_output(2, ctx->MakeShape(y_red));
    }
    else
    {
        ctx->set_output(0, ctx->UnknownShape());
        ctx->set_output(1, ctx->UnknownShape());
        ctx->set_output(2, ctx->UnknownShape());
    }
    return Status::OK();
}
```
