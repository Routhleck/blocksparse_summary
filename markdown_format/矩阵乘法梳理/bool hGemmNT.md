---
title: bool hGemmNT
date: 2023-03-20T15:54:34Z
lastmod: 2023-03-20T15:55:56Z
---

# bool hGemmNT

```cpp
bool hGemmNT(CUstream stream, const ehalf* A, const ehalf* B, float* C, uint M, uint N, uint K, uint blk_a, uint blk_b, uint blk_A, uint blk_B, uint accumulate, float scale)
{
    if (scale != 0.0f)
    {
        dim3 grid(blk_a*blk_b, blk_B, blk_A);
        if (M & 31)
            if (accumulate)
                hgemm_32x32x64_nt<false, true><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_32x32x64_nt<false,false><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
        else
            if (accumulate)
                hgemm_32x32x64_nt< true, true><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
            else
                hgemm_32x32x64_nt< true,false><<<grid,128,0,stream>>>(A, B, C, M, N, K, blk_a, blk_b, scale);
    }
    else if (accumulate == 0)
        cuMemsetD32Async((CUdeviceptr)C, 0, M*N, stream);

    return true;
}
```

‍
