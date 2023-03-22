---
title: explicit BlocksparseMatmulDGOp
date: 2023-03-20T16:16:31Z
lastmod: 2023-03-21T16:30:20Z
---

# explicit BlocksparseMatmulDGOp

只有一个ctx上下文参数，直接初始化

```cpp
explicit BlocksparseMatmulDGOp(OpKernelConstruction* ctx) : OpKernel(ctx) { }
```

‍
