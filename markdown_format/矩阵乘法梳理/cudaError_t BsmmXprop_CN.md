---
title: cudaError_t BsmmXprop_CN
date: 2023-03-20T15:35:49Z
lastmod: 2023-03-20T15:37:06Z
---

# cudaError_t BsmmXprop_CN

```cpp
template <bool Fprop, CTYPE(T)>
cudaError_t BsmmXprop_CN(const T* X, const T* W, T* Y, bsmm_params* params)
{
    dim3 grid(CEIL_DIV(params->N, 64), params->segments, 1);

    const int2* L2 = (const int2*)params->Lut;
    const   T2* W2 = (const   T2*)W;
    const   T4* W4 = (const   T4*)W;
    const   T4* X4 = (const   T4*)X;
    const   T8* X8 = (const   T8*)X;
            T2* Y2 = (        T2*)Y;
            T4* Y4 = (        T4*)Y;
            T8* Y8 = (        T8*)Y;

    if (params->locks > 0)
        cuMemsetD32Async((CUdeviceptr)params->Lock, 0, grid.x * params->locks * 2, params->stream);

    if (params->bsize == 8)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_08x64x08x8_xprop<Fprop,T2,T8,T8><<<grid,32,params->shared,params->stream>>>(L2, W2, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_08x64x08x4_xprop<Fprop,T2,T4,T2><<<grid,32,params->shared,params->stream>>>(L2, W2, X4, Y2, params->Lock, params->locks, params->N);
    }
    else if (params->bsize == 16)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_16x64x16x8_xprop<Fprop,T4,T8,T8><<<grid,64,params->shared,params->stream>>>(L2, W4, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_16x64x16x4_xprop<Fprop,T4,T4,T2><<<grid,64,params->shared,params->stream>>>(L2, W4, X4, Y2, params->Lock, params->locks, params->N);
    }
    else
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_32x64x32x8_xprop<Fprop,T4,T8,T8><<<grid,128,params->shared,params->stream>>>(L2, W4, X8, Y8, params->Lock, params->locks, params->N>>3);
        else
            gemm_blocksparse_32x64x32x4_xprop<Fprop,T4,T4,T4><<<grid,128,params->shared,params->stream>>>(L2, W4, X4, Y4, params->Lock, params->locks, params->N>>2);
    }
    return cudaPeekAtLastError();
}
template cudaError_t BsmmXprop_CN<true,  VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<true,  VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<true,  VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);

template cudaError_t BsmmXprop_CN<false, VTYPE(float)>(const float* X, const float* W, float* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<false, VTYPE(ehalf)>(const ehalf* X, const ehalf* W, ehalf* Y, bsmm_params* params);
template cudaError_t BsmmXprop_CN<false, VTYPE(bhalf)>(const bhalf* X, const bhalf* W, bhalf* Y, bsmm_params* params);
```

‍
