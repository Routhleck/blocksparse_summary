---
title: Status Compute_Xprop
date: 2023-03-20T16:00:23Z
lastmod: 2023-03-21T15:49:47Z
---

# Status Compute_Xprop

确定各种参数...在此过程用调用ClosestDivisorTo4  
最终计算时，确定是否有tensorcores

参考如下判断流程根据tensorcores、axis_和blk的值来分别调用不同的核函数

> hgemm_blocksparse_xn_64_sdd  
> hgemm_blocksparse_xn_128_sdd  
> hgemm_blocksparse_nxBsmmXprop_CN  
> BsmmXprop_CN

```mindmap
- 判断tensorcores
  - 有tensorcores
    - axis_ == 0
      - blk == 64
        - hgemm_blocksparse_xn_64_sdd
      - blk != 64
        - hgemm_blocksparse_xn_128_sdd
    - axis_ != 0
      - hgemm_blocksparse_nx_dsd
  - 无tensorcores
    - Gate == NULL且axis_ == 0
      - op == FPROP_OP
        - BsmmXprop_CN< true,NTYPE(T)>
      - op != BPROP_OP
        - BsmmXprop_CN<false,NTYPE(T)>
    - else
      - 报错(目前只支持fp16)
```

```cpp
/* tensorcores的判断
判断当前 GPU 是否支持 Tensor Cores，并且 T1 是否是 ehalf 类型（即 NVIDIA Tensor Cores 支持的半精度浮点类型）。
具体实现是通过检查当前 GPU 的主版本号是否大于等于 7 来判断是否支持 Tensor Cores。在 CUDA 9 中，NVIDIA 发布了 Volta 架构的 GPU，并加入了 Tensor Cores 的支持。这个架构的主版本号为 7，因此主版本号大于等于 7 的 GPU 都支持 Tensor Cores。
另外，代码还使用了 std::is_same 模板来判断类型是否相同。在这里，T1 的类型是否是 ehalf，即是否是半精度浮点类型，也是判断是否支持 Tensor Cores 的条件之一。
*/
bool tensorcores = major_ >= 7 && std::is_same<T1, ehalf>::value;
```

```cpp
Status Compute_Xprop(OpKernelContext* ctx, uint op)
    {
        const Tensor& A = ctx->input(0);
        const Tensor& B = ctx->input(1);
        const Tensor& L = ctx->input(2);

        OpInputList gate;
        ctx->input_list("gate", &gate);

        TensorShape shapeC;
        int N     = 1;
        int rankA = A.dims();
        for (int i = 0; i < rankA; i++)
            if (i != axis_)
            {
                shapeC.AddDim(A.dim_size(i));
                N *= A.dim_size(i);
            }
            else
                shapeC.AddDim(params_.K);

        bool tensorcores = major_ >= 7 && std::is_same<T1, ehalf>::value;

        int blkN = 128, gridN = CEIL_DIV(N, 128), modN128 = N & 127;
        if (!tensorcores || axis_ == 1 || (modN128 > 0 && modN128 <= 64) || gridN * params_.segments < SMs_*4)
        {
            blkN  = 64;
            gridN = CEIL_DIV(N, 64);
        }

        Tensor* C;
        Status s = ctx->allocate_output(0, shapeC, &C);
        if (!s.ok()) return s;

        Tensor* Lock;
        TensorShape shapeL;
        if (params_.locks > 0)
            shapeL.AddDim(gridN * params_.locks * 2);
        s = ctx->allocate_output(1, shapeL, &Lock);
        if (!s.ok()) return s;

        params_.Lock = params_.locks > 0 ? Lock->flat<int32>().data() : nullptr;
        params_.N    = N;
        params_.Lut  = (const int*)L.flat<int64>().data();
        params_.Gate = gate.size() > 0 ? gate[0].flat<float>().data() : NULL;

        if (params_.blk_A == 0)
        {
            ClosestDivisorTo4(params_.segments, true, &params_.blk_a, &params_.blk_A);
            ClosestDivisorTo4(gridN,           false, &params_.blk_b, &params_.blk_B);

            // printf("%d %d %d %d %d %d\n", params_.segments, gridN, params_.blk_a, params_.blk_b, params_.blk_A, params_.blk_B);
        }

        const T1* pA = (const T1*)A.flat<T>().data();
        const T1* pB = (const T1*)B.flat<T>().data();
              T1* pC = (      T1*)C->flat<T>().data();

        if (is_gpu_)
            params_.stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        Benchmark* bench = nullptr;
        if (bench_) bench = new Benchmark(params_.stream, bench_string_, 0, flops_ * params_.N * params_.pcount, repeat_, is_gpu_);

        cudaError_t res;
        for (int r = 0; r < repeat_; r++)
            if (tensorcores)
            {
                if (axis_ == 0)
                    if (blkN == 64)
                        res = hgemm_blocksparse_xn_64_sdd( pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                    else
                        res = hgemm_blocksparse_xn_128_sdd(pA, pB, pC, &params_, op == FPROP_OP ? OP_T : OP_N);
                else
                    res = hgemm_blocksparse_nx_dsd(pA, pB, pC, &params_, op == FPROP_OP ? OP_N : OP_T);
            }
            else
            {
                if (params_.Gate == NULL && axis_ == 0)
                {
                    if (op == FPROP_OP)
                        res = BsmmXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                    else
                        res = BsmmXprop_CN<false,NTYPE(T)>(pA, pB, pC, &params_);
                }
                else
                {
                    // Cuda update for Volta broke these kernels.  Need to fix.
                    // Ideally merge gated and non-gated code like is done with hgemm kernels.
                    return errors::Internal("Gated blocksparse matmul currently only supported on fp16 tensorcores.");
                    // if (op == NN_OP)
                    //     res = BsmmGatedXprop_CN<false,NTYPE(T)>(pA, pB, pC, &params_);
                    // else
                    //     res = BsmmGatedXprop_CN< true,NTYPE(T)>(pA, pB, pC, &params_);
                }
            }

        if (bench) delete bench;

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
```
