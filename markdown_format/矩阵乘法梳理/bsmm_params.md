---
title: bsmm_params
date: 2023-03-22T11:39:17Z
lastmod: 2023-03-22T14:24:21Z
---

# bsmm_params

|成员名|成员类型|成员说明|
| ----------| --------------| ---------------------------------|
|Lut|const int*|稀疏矩阵的索引表|
|Gate|const float*|激活函数的门限值?|
|Lock|int*|互斥锁数组|
|blocks|int|稀疏块的数量|
|bsize|int|稀疏块的尺寸|
|segments|int|数据分割的数量|
|locks|int|互斥锁的数量|
|C|int|卷积核的通道数|
|K|int|卷积核的数量|
|N|int|输出特征图的像素数|
|shared|int|共享内存的大小|
|pcount|int|不为0的元素数量? 好像默认值都=1|
|blk_a|uint|稠密矩阵块的大小|
|blk_A|uint|稠密矩阵行方向的步长|
|blk_b|uint|稠密矩阵块的大小|
|blk_B|uint|稠密矩阵列方向的步长|
|alpha|float|学习率|
|beta|float|权重衰减率|
|stream|CUstream|CUDA流对象|

```cpp
typedef struct bsmm_params
{
    const int* Lut;
    const float* Gate;
    int* Lock;
    //float4* Scratch;
    int blocks;
    int bsize;
    int segments;
    int locks;
    int C;
    int K;
    int N;
    int shared;
    int pcount;
    uint blk_a;
    uint blk_A;
    uint blk_b;
    uint blk_B;
    float alpha;
    float beta;
    CUstream stream;
} bsmm_params;
```

‍
