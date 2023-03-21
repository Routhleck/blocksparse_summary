---
title: bool IdentityInitCK
date: 2023-03-20T15:15:41Z
lastmod: 2023-03-20T15:52:27Z
---

# bool IdentityInitCK

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
