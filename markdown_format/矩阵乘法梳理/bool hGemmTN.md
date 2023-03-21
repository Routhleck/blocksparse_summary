---
title: bool hGemmTN
date: 2023-03-20T15:54:25Z
lastmod: 2023-03-20T15:55:43Z
---

# bool hGemmTN

```cpp
bool hGemmTN(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale)
{
    if (scale != 0.0f)
    {
        dim3 grid(blk_a*blk_b, blk_B, blk_A);
        if (M & 63)
            if (accumulate)
                hgemm_64x64x32_tn<false, true><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_64x64x32_tn<false,false><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
        else
            if (accumulate)
                hgemm_64x64x32_tn< true, true><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_64x64x32_tn< true,false><<<grid,256,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
    }
    else if (accumulate == 0)
        cuMemsetD32Async((CUdeviceptr)C, 0, M*N, stream);

    return true;
}
```
