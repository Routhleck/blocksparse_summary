---
title: void compute
date: 2023-03-20T16:00:05Z
lastmod: 2023-03-21T14:48:32Z
---

# void compute

判断OP来判断执行的是向前还是向后传递，分别调用Compute_Xprop和Compute_Updat

```cpp
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
```

‍
