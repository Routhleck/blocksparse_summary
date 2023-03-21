---
title: bool BlocksparseFeatureReduceNC
date: 2023-03-20T15:54:07Z
lastmod: 2023-03-20T15:55:09Z
---

# bool BlocksparseFeatureReduceNC

```cpp
bool BlocksparseFeatureReduceNC(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type)
{
    uint gridC   = C >> bshift;
    uint threads = params * 32;
    if (bshift == 5)
    {
        dim3 grid(gridC, CEIL_DIV(N, 32), 1);
        if (norm_type == MAX_NORM)
            blocksparse_feature_reduce_nc<32,32,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
        else
            blocksparse_feature_reduce_nc<32,32, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
    }
    else if (bshift == 6)
    {
        dim3 grid(gridC, CEIL_DIV(N, 16), 1);
        if (norm_type == MAX_NORM)
            blocksparse_feature_reduce_nc<64,16,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
        else
            blocksparse_feature_reduce_nc<64,16, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, N, C);
    }
    return true;
}
```
