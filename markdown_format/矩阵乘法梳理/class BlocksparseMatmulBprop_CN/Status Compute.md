---
title: Status Compute
date: 2023-03-22T15:08:03Z
lastmod: 2023-03-22T15:08:30Z
---

# Status Compute

参考如下判断流程根据tensorcores和Gate的值来分别调用不同的核函数

> hgemm_blocksparse_xn_sdd  
> cudaError_t BsmmXprop_CN  
> cudaError_t BsmmGatedXprop_CN

```mindmap
- 判断tensorcores
  - 有tensorcores
    - hgemm_blocksparse_xn_sdd
  - 无tensorcores
    - Gate == NULL
      - BsmmXprop_CN
    - Gate != NULL
      - BsmmGatedXprop_CN
```

```cpp
virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        if (this->major_ == 0)
            GetCountSMsVersion(&this->major_, NULL);

        cudaError_t res;
        if (this->major_ >= 7 && std::is_same<TA, ehalf>::value)
            res = hgemm_blocksparse_xn_sdd(A, B, C, this->params_, 0);
        else
            if (this->params_->Gate == NULL)
                res = BsmmXprop_CN<false,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
            else
                res = BsmmGatedXprop_CN<false,VTYPE3(TA,TB,TC)>(A, B, C, this->params_);

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
```

‍
