# pytorch_complex

[![Build Status](https://travis-ci.org/kamo-naoyuki/pytorch_complex.svg?branch=master)](https://travis-ci.org/kamo-naoyuki/pytorch_complex)
[![codecov](https://codecov.io/gh/kamo-naoyuki/pytorch_complex/branch/master/graph/badge.svg)](https://codecov.io/gh/kamo-naoyuki/pytorch_complex)

A temporal python class for PyTorch-ComplexTensor


## What is this?
A Python class to perform as `ComplexTensor` in PyTorch: Nothing except for the following,

```python
class ComplexTensor: 
    def __init__(self, ...):
        self.real = torch.Tensor(...)
        self.imag = torch.Tensor(...)
```

### Why?
PyTorch is great DNN Python library, except that it doesn't support `ComplexTensor` in Python level.

https://github.com/pytorch/pytorch/issues/755

I'm looking forward to the completion, but I need `ComplexTensor` for now.
 I created this cheap module for the temporal replacement of it. Thus, I'll throw away this project as soon as  `ComplexTensor` is completely supported!

## Requirements

```
Python>=3.6
PyTorch>=1.0
```

## Install

```
pip install git+https://github.com/kamo-naoyuki/pytorch_complex
```

## How to use

### Basic mathematical operation
```python
import numpy as np
from torch_complex.tensor import ComplexTensor

real = np.random.randn(3, 10, 10)
imag = np.random.randn(3, 10, 10)

x = ComplexTensor(real, imag)
x.numpy()

x + x
x * x
x - x
x / x
x ** 1.5
x @ x  # Batch-matmul
x.conj()
x.inverse() # Batch-inverse
```

All are implemented with combinations of computation of `RealTensor` in python level, thus the speed　is not good enough.


### Functional

```python
import torch_complex.functional as F
F.cat([x, x])
F.stack([x, x])
F.matmul(x, x)  # Same as x @ x
F.einsum('bij,bjk,bkl->bil', [x, x, x])
```

### For DNN
Almost all methods that `torch.Tensor` has are implemented. 

```python
x.cuda()
x.cpu()
(x + x).sum().backward()
```
