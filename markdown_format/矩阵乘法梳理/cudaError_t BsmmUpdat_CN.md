---
title: cudaError_t BsmmUpdat_CN
date: 2023-03-20T15:37:27Z
lastmod: 2023-03-20T15:37:36Z
---

# cudaError_t BsmmUpdat_CN

```cpp
template <CTYPE(T)>
cudaError_t BsmmUpdat_CN(const T* X, const T* E, T* U, bsmm_params* params)
{
    dim3 grid(params->blocks, 1, 1);
    int loops = CEIL_DIV(params->N, 64);

    struct Plist<T4,8>* X4 = (struct Plist<T4,8>*)X;
    struct Plist<T4,8>* E4 = (struct Plist<T4,8>*)E;
    struct Plist<T8,8>* X8 = (struct Plist<T8,8>*)X;
    struct Plist<T8,8>* E8 = (struct Plist<T8,8>*)E;

    const int2* L2 = (const int2*)params->Lut;
            T2* U2 = (        T2*)U;
            T4* U4 = (        T4*)U;

    if (params->bsize == 8)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_08x64x08x8_updat<T8,T8,T2><<<grid,32,0,params->stream>>>(*X8, *E8, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_08x64x08x4_updat<T4,T4,T2><<<grid,32,0,params->stream>>>(*X4, *E4, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    else if (params->bsize == 16)
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_16x64x16x8_updat<T8,T8,T4><<<grid,64,0,params->stream>>>(*X8, *E8, L2, U4, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_16x64x16x4_updat<T4,T4,T4><<<grid,64,0,params->stream>>>(*X4, *E4, L2, U4, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    else
    {
        if (sizeof(T) == 2 && (params->N & 7) == 0)
            gemm_blocksparse_32x64x32x8_updat<T8,T8,T2><<<grid,256,0,params->stream>>>(*X8, *E8, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
        else
            gemm_blocksparse_32x64x32x4_updat<T4,T4,T2><<<grid,256,0,params->stream>>>(*X4, *E4, L2, U2, params->pcount*8, params->N, loops, params->alpha, params->beta);
    }
    return cudaPeekAtLastError();
}
template cudaError_t BsmmUpdat_CN<VTYPE(float)>(const float* X, const float* E, float* U, bsmm_params* params);
template cudaError_t BsmmUpdat_CN<VTYPE(ehalf)>(const ehalf* X, const ehalf* E, ehalf* U, bsmm_params* params);
template cudaError_t BsmmUpdat_CN<VTYPE(bhalf)>(const bhalf* X, const bhalf* E, bhalf* U, bsmm_params* params);

```

‍
