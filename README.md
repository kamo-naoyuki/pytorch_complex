# pytorch_complex

## What is this?
A Python class to perform `ComplexTensor` for PyTorch: Nothing except for the following,

```python
class ComplexTensor: 
    def __init__(self, ...):
        self.real = torch.Tensor(...)
        self.imag = torch.Tensor(...)
```

### Why need?
I think PyTorch is greatest DNN Python library now, except that it doesn't support `ComplexTensor` in Python level.

https://github.com/pytorch/pytorch/issues/755

I'm looking forward to the completion, but I need `ComplexTensor` for some DNN experiments now and I don't want to use any other DNN tools.
 I created this cheap module for the workaround of this problem.

I'll throw away this project as soon as  `ComplexTensor` is completely supported!

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

All is implemented with combinations of python-function of `RealTensor`, thus the computing speed　is not good enough.


### Functional

```python
import torch_complex.functional as F
F.cat([x, x])
F.stack([x, x])
F.matmul(x, x)  # Same as x @ x
F.einsum('bij,bjk,bkl->bil', [x, x, x])
```

### For DNN
Almost all method that `torch.Tensor` has is implemented. If that doesn't exist, it it easy to implement.

```python
(x + x).sum().backward()
```
