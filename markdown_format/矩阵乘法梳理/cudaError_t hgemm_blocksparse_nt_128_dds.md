---
title: cudaError_t hgemm_blocksparse_nt_128_dds
date: 2023-03-20T15:49:24Z
lastmod: 2023-03-20T15:49:36Z
---

# cudaError_t hgemm_blocksparse_nt_128_dds

```cpp
cudaError_t hgemm_blocksparse_nt_128_dds(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params)
{
    struct Plist<ehalf,8>* X8 = (struct Plist<ehalf,8>*)X;
    struct Plist<ehalf,8>* E8 = (struct Plist<ehalf,8>*)E;

    const uint2* Lut = (const uint2*)params->Lut;
    uint accumulate  = params->beta == 1.0f;
    uint pcount8     = params->pcount * 8;
    uint N           = params->N;
    uint loops       = CEIL_DIV(N, 128);
    bool k128        = (N & 127) == 0;

    dim3 grid(params->blocks, 1, 1);

    if (params->bsize == 8)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_8x8x128_nt_dds< true,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x128_nt_dds<false,false><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_8x8x128_nt_dds< true, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_8x8x128_nt_dds<false, true><<<grid,64,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_16x16x128_nt_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x128_nt_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_16x16x128_nt_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_16x16x128_nt_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    else if (params->bsize == 32)
    {
        if (params->Gate == 0)
        {
            if (k128)
                hgemm_blocksparse_32x32x128_nt_dds< true,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x128_nt_dds<false,false><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
        else
        {
            if (k128)
                hgemm_blocksparse_32x32x128_nt_dds< true, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
            else
                hgemm_blocksparse_32x32x128_nt_dds<false, true><<<grid,128,0,params->stream>>>(*X8, *E8, U, Lut, params->Gate, pcount8, N, loops, accumulate);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_nt_128_dds(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_nt_128_dds(const float* X, const float* E, float* U, bsmm_params* params) { return cudaSuccess; }
```
