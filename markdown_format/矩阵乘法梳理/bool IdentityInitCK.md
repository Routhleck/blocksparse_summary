---
title: bool IdentityInitCK
date: 2023-03-20T15:15:41Z
lastmod: 2023-03-21T17:14:27Z
---

# bool IdentityInitCK

根据 `bsize`​ 的不同值调用不同的 CUDA 内核函数 `identity_init_CK`​，以生成不同大小的方阵权重矩阵。这些内核函数的具体实现是在另外的代码文件中定义的，例如 `identity_init_CK<8,32>`​ 表示调用模板函数 `identity_init_CK`​ 生成大小为 8x8 的方阵权重矩阵，并且每个线程块（block）有 32 个线程。

```cpp
bool IdentityInitCK(CUstream stream, float* W, const int* lut, int CB, int KB, int blocks, int bsize, float scale)
{
         if (bsize ==  8)
        identity_init_CK< 8,  32><<<blocks,  32, 0, stream>>>(W, (const int2*)lut, CB, KB, scale);
    else if (bsize == 16)
        identity_init_CK<16,  64><<<blocks,  64, 0, stream>>>(W, (const int2*)lut, CB, KB, scale);
    else if (bsize == 32)
        identity_init_CK<32, 256><<<blocks, 256, 0, stream>>>(W, (const int2*)lut, CB, KB, scale);
    else if (bsize == 64)
        identity_init_CK<64,1024><<<blocks,1024, 0, stream>>>(W, (const int2*)lut, CB, KB, scale);
    return true; // TODO
}
```

‍
