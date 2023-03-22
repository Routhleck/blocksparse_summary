---
title: bool BlocksparseGateGrad
date: 2023-03-20T15:51:19Z
lastmod: 2023-03-21T16:55:12Z
---

# bool BlocksparseGateGrad

根据不同的bsize来使用不同的线程数调用blocksparse_gate_grad(blocksparse_hgemm_cn_64_op_gpu.cu)

```cpp
template <typename T>
bool BlocksparseGateGrad(CUstream stream, T* dw_out, float* dg, const T* dw, const T* w, const float* g, uint blocks, uint bsize)
{
         if (bsize ==  8)
        blocksparse_gate_grad<T, 8,  32><<<blocks,  32,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 16)
        blocksparse_gate_grad<T,16,  64><<<blocks,  64,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 32)
        blocksparse_gate_grad<T,32, 256><<<blocks, 256,0,stream>>>(dw_out, dg, dw, w, g);
    else if (bsize == 64)
        blocksparse_gate_grad<T,64,1024><<<blocks,1024,0,stream>>>(dw_out, dg, dw, w, g);
    return true;
}
template bool BlocksparseGateGrad<float>(CUstream stream, float* dw_out, float* dg, const float* dw, const float* w, const float* g, uint blocks, uint bsize);
template bool BlocksparseGateGrad<ehalf>(CUstream stream, ehalf* dw_out, float* dg, const ehalf* dw, const ehalf* w, const float* g, uint blocks, uint bsize);
```

‍
