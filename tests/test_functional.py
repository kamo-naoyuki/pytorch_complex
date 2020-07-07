from distutils.version import LooseVersion

import numpy
import pytest

import torch
import torch_complex.functional as F
from torch_complex.tensor import ComplexTensor


def _get_complex_array(*shape):
    return numpy.random.randn(*shape) + 1j + numpy.random.randn(*shape)


@pytest.mark.parametrize(
    "nop,top",
    [
        (numpy.concatenate, F.cat),
        (numpy.stack, F.stack),
        (
            lambda x: numpy.einsum("ai,ij,jk->ak", *x),
            lambda x: F.einsum("ai,ij,jk->ak", x),
        ),
    ],
)
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
    t = ComplexTensor(_get_complex_array(10, 10))
    x = numpy.trace(t.numpy())
    y = F.trace(t).numpy()
    numpy.testing.assert_allclose(x, y)


@pytest.mark.skipif(
    LooseVersion(torch.__version__) <= LooseVersion("1.0"), reason="requires torch>=1.1"
)
def test_solve():
    t = ComplexTensor(_get_complex_array(1, 10, 10))
    s = ComplexTensor(_get_complex_array(1, 10, 4))
    x, _ = F.solve(s, t)
    y = t @ x
    numpy.testing.assert_allclose(
        y.real.numpy()[0], s.real.numpy()[0], atol=1e-13,
    )
    numpy.testing.assert_allclose(
        y.imag.numpy()[0], s.imag.numpy()[0], atol=1e-13,
    )
