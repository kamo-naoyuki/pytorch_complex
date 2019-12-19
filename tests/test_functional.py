import numpy
import pytest

import torch_complex.functional as F
from torch_complex.tensor import ComplexTensor


def _get_complex_array(*shape):
    return numpy.random.randn(*shape) + 1j + numpy.random.randn(*shape)


@pytest.mark.parametrize('nop,top',
                         [(numpy.concatenate, F.cat),
                          (numpy.stack, F.stack),
                          (lambda x: numpy.einsum('ai,ij,jk->ak', *x),
                           lambda x: F.einsum('ai,ij,jk->ak', x))])
def test_operation(nop, top):
    if top is None:
        top = nop
    n1 = _get_complex_array(10, 10)
    n2 = _get_complex_array(10, 10)
    n3 = _get_complex_array(10, 10)
    t1 = ComplexTensor(n1.copy())
    t2 = ComplexTensor(n2.copy())
    t3 = ComplexTensor(n3.copy())

    x = nop([n1, n2, n3])
    y = top([t1, t2, t3])
    y = y.numpy()
    numpy.testing.assert_allclose(x, y)


def test_trace():
    t = _get_complex_array(10, 10)
    x = numpy.trace(t.numpy())
    y = t.trace().numpy()
    numpy.testing.assert_allclose(x, y)
