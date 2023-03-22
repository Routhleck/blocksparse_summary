---
title: BlocksparseMatmul
date: 2023-03-22T11:34:35Z
lastmod: 2023-03-22T14:54:24Z
---

# BlocksparseMatmul

构造函数

# 输入参数

|参数|类型|说明|
| --------| -------------| ---------------------|
|params|bsmm_params|矩阵乘法参数|
|major_|int|CUDA主版本，默认为0|

```cpp
BlocksparseMatmul(bsmm_params* params) : params_(params), major_(0) {}
```

‍
