---
title: cudaError_t BsmmGatedXprop_CN
date: 2023-03-20T15:42:58Z
lastmod: 2023-03-20T15:43:11Z
---

# cudaError_t BsmmGatedXprop_CN

```cpp
template <bool Fprop, CTYPE(T)>
cudaError_t BsmmGatedXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params)
{
    dim3 grid(CEIL_DIV(params->N, 64), params->segments, 1);

    // printf("grid: %d %d\n", grid.x, grid.y);

    const int2* L2 = (const int2*)params->Lut;
    const   T2* W2 = (const   T2*)W;
    const   T4* X4 = (const   T4*)X;
    const   T8* X8 = (const   T8*)X;
            T2* Y2 = (        T2*)Y;
            T8* Y8 = (        T8*)Y;

    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, grid.x * params->locks * 2, params->stream);

    if (params->bsize == 8)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_gated_08x64x08x8_xprop<Fprop,T2,T8,T8><<<grid,32,params->shared*2,params->stream>>>(L2, params->Gate, W2, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_gated_08x64x08x4_xprop<Fprop,T2,T4,T2><<<grid,32,params->shared*2,params->stream>>>(L2, params->Gate, W2, X4, Y2, params->Lock, params->locks, params->N);
    }
    return cudaPeekAtLastError();
}
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<true,  VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);

template cudaError_t BsmmGatedXprop_CN<false, VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<false, VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmGatedXprop_CN<false, VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);
```
