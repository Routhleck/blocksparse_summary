---
title: Status Compute
date: 2023-03-22T15:09:02Z
lastmod: 2023-03-22T15:12:07Z
---

# Status Compute

参考如下判断流程根据tensorcores和Gate的值来分别调用不同的核函数

> hgemm_blocksparse_nt_dds  
> cudaError_t BsmmUpdat_CN  
> cudaError_t BsmmGatedUpdat_CN

```mindmap
- 判断tensorcores
  - 有tensorcores
    - hgemm_blocksparse_nt_dds
  - 无tensorcores
    - Gate == NULL
      - BsmmUpdat_CN
    - Gate != NULL
      - BsmmGatedUpdat_CN
```

```cpp
virtual Status Compute(const TA* A, const TB* B, TC* C)
    {
        if (this->major_ == 0)
            GetCountSMsVersion(&this->major_, NULL);

        cudaError_t res;
        if (this->major_ >= 7 && std::is_same<TA, ehalf>::value)
            res = hgemm_blocksparse_nt_dds(A, B, C, this->params_);
        else
            if (this->params_->Gate == NULL)
                res = BsmmUpdat_CN<VTYPE3(TA,TB,TC)>(A, B, C, this->params_);
            else
                res = BsmmGatedUpdat_CN<VTYPE3(TA,TB,TC)>(A, B, C, this->params_);

        if (cudaSuccess != res)
            return errors::Internal(cudaGetErrorString(res));
        return Status::OK();
    }
```
