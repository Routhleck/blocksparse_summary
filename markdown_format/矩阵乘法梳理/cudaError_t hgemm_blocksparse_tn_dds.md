---
title: cudaError_t hgemm_blocksparse_tn_dds
date: 2023-03-20T15:50:20Z
lastmod: 2023-03-20T15:50:44Z
---

# cudaError_t hgemm_blocksparse_tn_dds

```cpp
cudaError_t hgemm_blocksparse_tn_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params)
{
    struct Plist<ehalf,8>* X8 = (struct Plist<ehalf,8>*)X;
    struct Plist<ehalf,8>* E8 = (struct Plist<ehalf,8>*)E;

    const uint2* Lut = (const uint2*)params->Lut;
    uint accumulate  = params->beta == 1.0f;
    uint pcount8     = params->pcount * 8;
    uint N           = params->N;
    uint C           = params->C;
    uint K           = params->K;
    uint loops       = CEIL_DIV(N, 64);
    bool N64         = (N & 63) == 0;

    dim3 grid(params->blocks, 1, 1);

    if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (N64)
                hgemm_blocksparse_32x32x64_tn_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_tn_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
        else
        {
            if (N64)
                hgemm_blocksparse_32x32x64_tn_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_32x32x64_tn_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
    }
    else if (params->bsize == 64)
    {
        if (params->Gate == 0)
        {
            if (N64)
                hgemm_blocksparse_64x64x64_tn_dds< true,false><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_64x64x64_tn_dds<false,false><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
        else
        {
            if (N64)
                hgemm_blocksparse_64x64x64_tn_dds< true, true><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
            else
                hgemm_blocksparse_64x64x64_tn_dds<false, true><<<grid,256,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, C, K, loops, accumulate);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_tn_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_tn_dds(const float* X, const float* E, float* U, bsmm_params* params) { return cudaSuccess; }
```

‍
