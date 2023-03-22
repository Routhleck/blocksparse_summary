---
title: class BlocksparseMatmulOp
date: 2023-03-20T15:12:15Z
lastmod: 2023-03-22T14:56:37Z
---

# class BlocksparseMatmulOp

主要为前向、后向传播与更新的计算

# 成员方法

|成员方法|返回类型|说明|
| ----------------------| ----------| ---------------------------------------------------------------|
|BlocksparseMatmulOp|构造函数|构造函数，用于创建 BlocksparseMatmulOp 对象|
|Compute|void|重载 OpKernel 的 Compute 函数，用于执行前向传播或反向传播操作|
|Status Compute_Xprop|Status|传播函数，执行稀疏矩阵前向或后向传播操作|
|Status Compute_Updat|Status|更新函数，执行稀疏矩阵乘法的更新操作|

# 成员变量

|成员变量 |变量类型 |说明 |
| ----------------------------| ----------------------------------------------------------------| -------|
|​​params_​​|bsmm_params|稀疏矩阵乘法的参数|
|​axis_​​|int|用于指定进行矩阵乘法运算的维度以及计算张量 `C`​​ 的输出形状。<br />|
|​​bench_​​|int|benchmark循环次数|
|​​repeat_​​|int|向前传递、向后传递次数|
|​​SMs_​​|int|SM核心数量|
|​​major_​​|int|CUDA主版本号|
|​​~~grid_n_~~​​|~~int~~|~~CUDA网格数量？没有用到~~|
|​​flops_​​|float|用于记录浮点运算次数，用于性能评测|
|​​gated_dw_​​|bool|用于记录是否进行 gated 操作|
|​​is_gpu_​​|bool|是否有GPU|
|​​bench_string_[256]​​|char|benchmark名字|

# 具体代码

```cpp
template <uint OP, MTYPE(T)>
class BlocksparseMatmulOp : public OpKernel
{
public:
    explicit BlocksparseMatmulOp(OpKernelConstruction* ctx) : OpKernel(ctx), SMs_(0), major_(0), repeat_(1), flops_(0.0f)
    {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("segments", &params_.segments));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("locks",    &params_.locks   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("blocks",   &params_.blocks  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bsize",    &params_.bsize  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("C",        &params_.C       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("K",        &params_.K       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("shared",   &params_.shared  ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha",    &params_.alpha   ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta",     &params_.beta    ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("gated_dw", &gated_dw_       ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("axis",     &axis_ ));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("bench",    &bench_));
        params_.pcount = 1;
        params_.blk_A  = 0;

        is_gpu_ = ctx->device_type() == DEVICE_GPU;

        //OP_REQUIRES(ctx, axis_ == 0, errors::InvalidArgument("Only feature axis=0 currently supported."));

        // TODO: pack larger values of K in gridZ
        OP_REQUIRES(ctx, params_.K < params_.bsize*65536, errors::InvalidArgument("K < bsize*65536"));
        OP_REQUIRES(ctx, params_.C < params_.bsize*65536, errors::InvalidArgument("C < bsize*65536"));

        if (bench_)
        {
            repeat_ = bench_;
            flops_  = (float)(params_.blocks * params_.bsize*params_.bsize);

            const char* op = OP == FPROP_OP ? "FPROP" : OP == BPROP_OP ? "BPROP" : "UPDAT";
            sprintf(bench_string_, "%s %02d-%d C:%05d K:%05d blks:%d", op, params_.bsize, axis_, params_.C, params_.K, params_.blocks);
        }
    }
    void Compute(OpKernelContext* ctx) override
    {
        if (major_ == 0)
        {
            SMs_ = GetCountSMsVersion(&major_, NULL);
            //OP_REQUIRES(ctx, major_ >= 7, errors::InvalidArgument("Tensorcore GPU required"));
        }
        if (OP == UPDAT_OP)
            OP_REQUIRES_OK(ctx, this->Compute_Updat(ctx));
        else
            OP_REQUIRES_OK(ctx, this->Compute_Xprop(ctx, OP));
    }
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
    bsmm_params params_;
    int   axis_, bench_, repeat_, SMs_, major_, grid_n_;
    float flops_;
    bool  gated_dw_, is_gpu_;
    char  bench_string_[256];
};
```
