---
title: BlocksparseMatmul_NC
date: 2023-03-22T14:33:57Z
lastmod: 2023-03-22T14:36:56Z
---

# BlocksparseMatmul_NC

构造函数,先调用基类的构造函数，对成员变量 `params_`​ 进行初始化。然后根据输入参数 `op`​、`depth`​ 和数据类型，计算出对应的 CUDA kernel 名称，并将其保存在类成员变量 `kernel_name_`​ 中。最后，根据 CUDA kernel 名称和 CUDA context，获取对应的 CUDA kernel 并保存在类成员变量 `kernel_`​ 中。

```cpp
BlocksparseMatmul_NC(bsmm_params* params, const char* op, int depth, int threads) :
        BlocksparseMatmul<VTYPE3(TA,TB,TC)>(params), threads_(threads)
    {
        const char* dtypeA = std::is_same<TA, ehalf>::value ? "A10" : std::is_same<TA, bhalf>::value ? "A7": "A32";
        const char* dtypeB = std::is_same<TB, ehalf>::value ? "B10" : std::is_same<TB, bhalf>::value ? "B7": "B32";
        const char* dtypeC = std::is_same<TC, ehalf>::value ? "C10" : std::is_same<TC, bhalf>::value ? "C7": "C32";

        // int depth;
        // const char* op;
        // if      (mode_  == 0) { op = "fprop"; depth = 32; threads_ = 128; }
        // else if (mode_  == 1) { op = "bprop"; depth = 32; threads_ = 128; }
        // else                  { op = "updat"; depth =  8; threads_ =  32; }

        char kernel_name[48];
        sprintf(kernel_name, "gemm_blocksparse_32x32x%d_%s_%s_%s_%s", depth, op, dtypeA, dtypeB, dtypeC);
        kernel_name_ = kernel_name;
        kernel_ = 0;
    }
```

‍
