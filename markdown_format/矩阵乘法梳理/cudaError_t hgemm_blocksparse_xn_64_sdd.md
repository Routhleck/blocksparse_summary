---
title: cudaError_t hgemm_blocksparse_xn_64_sdd
date: 2023-03-20T15:44:46Z
lastmod: 2023-03-20T15:45:03Z
---

# cudaError_t hgemm_blocksparse_xn_64_sdd

```cpp
cudaError_t hgemm_blocksparse_xn_64_sdd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op)
{
    dim3 grid(params->blk_a*params->blk_b, params->blk_B, params->blk_A);
    uint blk_N = params->blk_b * params->blk_B;

    //cuMemsetD16Async((CUdeviceptr)Y, 0, params->K * params->N, params->stream);
    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, blk_N * params->locks * 2, params->stream);

    const uint2* Lut = (const uint2*)params->Lut;
    uint* Lock       = (uint*)params->Lock;

    uint shared = params->shared + params->shared/2;

    if (params->bsize == 8)
    {
        shared += 4;
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_8x64x8_xn_sdd<OP_N,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x64x8_xn_sdd<OP_T,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_8x64x8_xn_sdd<OP_N, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_8x64x8_xn_sdd<OP_T, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 16)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_16x64x16_xn_sdd<OP_N,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x64x16_xn_sdd<OP_T,false><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_16x64x16_xn_sdd<OP_N, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_16x64x16_xn_sdd<OP_T, true><<<grid,64,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 32)
    {
        // 256 = (128+4)*16*4 - (64+16 + 32+16)*32*2
        shared = shared > 256 ? shared - 256 : 0;
        if (params->Gate == 0)
        {
            if (op == OP_N)
                hgemm_blocksparse_32x64x32_xn_sdd<OP_N,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x64x32_xn_sdd<OP_T,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                hgemm_blocksparse_32x64x32_xn_sdd<OP_N, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
            else
                hgemm_blocksparse_32x64x32_xn_sdd<OP_T, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, W, X, Y, Lock, params->locks, params->N, params->blk_a, params->blk_b, blk_N);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_xn_64_sdd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_xn_64_sdd(const float* X, const float* W, float* Y, bsmm_params* params, uint op) { return cudaSuccess; }
```
