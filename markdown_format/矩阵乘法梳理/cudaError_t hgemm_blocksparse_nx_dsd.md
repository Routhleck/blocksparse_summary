---
title: cudaError_t hgemm_blocksparse_nx_dsd
date: 2023-03-20T15:49:57Z
lastmod: 2023-03-20T15:50:17Z
---

# cudaError_t hgemm_blocksparse_nx_dsd

```cpp
cudaError_t hgemm_blocksparse_nx_dsd(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params, uint op)
{
    dim3 grid(params->blk_a*params->blk_b, params->blk_B, params->blk_A);
    uint blk_N = params->blk_b * params->blk_B;

    // cuMemsetD16Async((CUdeviceptr)Y, 0, params->K * params->N, params->stream);
    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, blk_N * params->locks * 2, params->stream);

    const uint2* Lut = (const uint2*)params->Lut;
    uint* Lock       = (uint*)params->Lock;

    bool   N64 = (params->N & 63) == 0;
    int shared = params->shared + params->shared/2;

    if (params->bsize == 32)
    {
        // 132*16*4 - ((32+8)*64 + (32+16)*32)*2
        shared -= op == OP_N ? 256 : 768;
        if (shared < 0) shared = 0;

        if (params->Gate == 0)
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N, true,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N,false,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T, true,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T,false,false><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N, true, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_N,false, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T, true, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x32x32_nx_dsd<OP_T,false, true><<<grid,128,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
    }
    else if (params->bsize == 64)
    {
        if (params->Gate == 0)
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N, true,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N,false,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T, true,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T,false,false><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
        else
        {
            if (op == OP_N)
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N, true, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_N,false, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
            else
                if (N64)
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T, true, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
                else
                    hgemm_blocksparse_64x64x64_nx_dsd<OP_T,false, true><<<grid,256,shared,params->stream>>>(Lut, params->Gate, X, W, Y, Lock, params->locks, params->N, params->C, params->K, params->blk_a, params->blk_b, blk_N);
        }
    }
    return cudaPeekAtLastError();
}
cudaError_t hgemm_blocksparse_nx_dsd(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params, uint op) { return cudaSuccess; }
cudaError_t hgemm_blocksparse_nx_dsd(const float* X, const float* W, float* Y, bsmm_params* params, uint op) { return cudaSuccess; }
```
