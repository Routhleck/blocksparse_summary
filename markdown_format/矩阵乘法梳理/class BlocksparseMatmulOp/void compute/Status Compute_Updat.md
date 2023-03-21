---
title: Status Compute_Updat
date: 2023-03-20T16:00:34Z
lastmod: 2023-03-21T15:52:50Z
---

# Status Compute_Updat

与Compute_Xprop类似  
先是各种判断来确定参数

参考如下判断流程根据tensorcores、axis_和blk的值来分别调用不同的核函数

> hgemm_blocksparse_nt_64_dds  
> hgemm_blocksparse_nt_128_dds  
> hgemm_blocksparse_tn_dds  
> BsmmUpdat_CN

```mindmap
- 判断tensorcores
  - 有tensorcores
    - axis_ == 0
      - modN128 > 0 && modN128 <= 64
        - hgemm_blocksparse_nt_64_dds
      - else
        - hgemm_blocksparse_nt_128_dds
    - axis_ != 0
      - hgemm_blocksparse_tn_dds
  - 无tensorcores
    - Gate == NULL且axis_ == 0
      - BsmmUpdat_CN<NTYPE(T)>
    - else
      - 报错(目前只支持fp16)
```

```cpp
Status Compute_Updat(OpKernelContext* ctx)
    {
        OpInputList x, dy, gate;

        ctx->input_list(   "x", &x);
        ctx->input_list(  "dy", &dy);
        ctx->input_list("gate", &gate);

        params_.pcount = x.size();

        if (params_.pcount > 8)
            return errors::Internal("No more than 8 inputs allowed.");

        struct Plist<T1,8> X;
        struct Plist<T1,8> DY;
        for (int i = 0; i < params_.pcount; ++i)
        {
             X.a[i] = (const T1*) x[i].flat<T>().data();
            DY.a[i] = (const T1*)dy[i].flat<T>().data();
        }
        params_.N = 1;
        int rank = x[0].dims();
        for (int i = 0; i < rank; i++)
            if (i != axis_)
                params_.N *= x[0].dim_size(i);

        T1* DW;
        if (params_.beta == 0.0f)
        {
            // BlocksparseMatmulDW: [x], [dy], lut, [gate]
            if (ctx->num_inputs() != params_.pcount*2 + 1 + gate.size())
                return errors::Internal("with beta=0.0, use BlocksparseMatmulDW ", ctx->num_inputs());

            Tensor* C;
            TensorShape shapeC({ params_.blocks, params_.bsize, params_.bsize });
            Status s = ctx->allocate_output(0, shapeC, &C);
            if (!s.ok()) return s;
            DW = (T1*)C->flat<T>().data();
        }
        else
        {
            // BlocksparseMatmulDWA: [x], [dy], lut, dwi, [gate]
            if (ctx->num_inputs() != params_.pcount*2 + 2 + gate.size())
                return errors::Internal("with beta!=0.0, use BlocksparseMatmulDWA ", ctx->num_inputs());

            // accumulate to C in place
            const Tensor& C = ctx->input(params_.pcount*2 + 1);
            ctx->set_output(0, C);
            DW = (T1*)C.flat<T>().data();
        }
        params_.Lut  = (const int*)ctx->input(params_.pcount*2).flat<int64>().data();
        params_.Gate = gated_dw_ && gate.size() > 0 ? gate[0].flat<float>().data() : NULL;

        if (is_gpu_)
            params_.stream = ((CUDAStream*)ctx->op_device_context()->stream()->implementation())->cuda_stream();

        Benchmark* bench = nullptr;
        if (bench_) bench = new Benchmark(params_.stream, bench_string_, 0, flops_ * params_.N * params_.pcount, repeat_, is_gpu_);

        cudaError_t res;
        for (int r = 0; r < repeat_; r++)
            if (major_ >= 7 && std::is_same<T1, ehalf>::value)
            {
                if (axis_ == 0)
                {
                    int modN128 = params_.N & 127;
                    if (modN128 > 0 && modN128 <= 64)
                        res = hgemm_blocksparse_nt_64_dds( (const T1*)&X, (const T1*)&DY, DW, &params_);
                    else
                        res = hgemm_blocksparse_nt_128_dds((const T1*)&X, (const T1*)&DY, DW, &params_);
                }
                else
                    res = hgemm_blocksparse_tn_dds((const T1*)&X, (const T1*)&DY, DW, &params_);
            }
            else
            {
                if (params_.Gate == NULL && axis_ == 0)
                    res = BsmmUpdat_CN<NTYPE(T)>((const T1*)&X, (const T1*)&DY, DW, &params_);
                else
                    return errors::Internal("Gated blocksparse matmul currently only supported on fp16 tensorcores.");
                    // res = BsmmGatedUpdat_CN<NTYPE(T)>((const T1*)&X, (const T1*)&DY, DW, &params_);
            }

        if (bench) delete bench;

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
```

‍
