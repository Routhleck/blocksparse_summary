---
title: bool BlocksparseFeatureReduceCN
date: 2023-03-20T15:54:13Z
lastmod: 2023-03-20T15:55:31Z
---

# bool BlocksparseFeatureReduceCN

```cpp
bool BlocksparseFeatureReduceCN(CUstream stream, ehalf* Y, const struct Plist<ehalf,8>* X8, uint params, uint C, uint N, uint bshift, uint norm_type)
{
    dim3 grid(CEIL_DIV(N, 64), C >> bshift, 1);
    uint threads = params * 32;

    if (norm_type == MAX_NORM)
    {
        if (bshift == 3)
            blocksparse_feature_reduce_cn< 8,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else if (bshift == 4)
            blocksparse_feature_reduce_cn<16,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else
            blocksparse_feature_reduce_cn<32,MAX_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
    }
    else
    {
        if (bshift == 3)
            blocksparse_feature_reduce_cn< 8, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else if (bshift == 4)
            blocksparse_feature_reduce_cn<16, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
        else
            blocksparse_feature_reduce_cn<32, L2_NORM><<<grid,threads,0,stream>>>(*X8, Y, params, N);
    }
    return true;
}
```
