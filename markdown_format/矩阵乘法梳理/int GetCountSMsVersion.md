---
title: int GetCountSMsVersion
date: 2023-03-22T14:20:45Z
lastmod: 2023-03-22T14:22:14Z
---

# int GetCountSMsVersion

```cpp
int GetCountSMsVersion(int* major, int* minor)
{
    CUdevice device; int count;
    cuCtxGetDevice(&device);
    cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    if (major != NULL)
        cuDeviceGetAttribute(major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    if (minor != NULL)
        cuDeviceGetAttribute(minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    return count;
}
```

‍
