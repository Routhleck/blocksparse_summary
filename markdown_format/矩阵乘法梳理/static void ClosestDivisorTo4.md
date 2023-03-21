---
title: static void ClosestDivisorTo4
date: 2023-03-20T15:09:54Z
lastmod: 2023-03-21T14:54:31Z
---

# static void ClosestDivisorTo4

找到一个最接近某个数的能被 4 整除的数，同时返回这个数除以这个最接近数的商和余数。**确定blk_a、blk_A、blk_b、blk_B**

具体实现是先判断这个数是否能被 4 整除，如果可以则直接返回 4 和商；如果不行，则按照一定的顺序依次尝试除以  3、5、2、7，找到能整除的数字并返回对应的商和除数。如果都不能整除，那么如果这个数是被用来计算矩阵乘法中的矩阵  A，那么就把这个数作为除数，商为 1；否则就把这个数作为商，除数为 1。

```cpp
static void ClosestDivisorTo4(uint val, bool isA, uint* div, uint* res)
{
         if ((val % 4) == 0) { *div = 4; *res = val / 4; }
    else if ((val % 3) == 0) { *div = 3; *res = val / 3; }
    else if ((val % 5) == 0) { *div = 5; *res = val / 5; }
    else if ((val % 2) == 0) { *div = 2; *res = val / 2; }
    else if ((val % 7) == 0) { *div = 7; *res = val / 7; }
    else if (isA) { *div = val; *res =   1; }
    else          { *div = 1;   *res = val; }
}
```

‍
