import numpy as np
import mytorch.tensor as mytorch

DEFAULT_DTYPE = mytorch.float32


def tensor(data, dtype=None, requires_grad=False):
    if isinstance(data, mytorch.Tensor):
        data = data.numpy()
    return mytorch.Tensor(
        data=np.array(data),
        dtype=dtype,
        requires_grad=requires_grad
    )


# TODO: Dynamically generate these functions


def zeros(*size, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.zeros(size),
        dtype=dtype,
        requires_grad=requires_grad
    )


def ones(*size, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.ones(size),
        dtype=dtype,
        requires_grad=requires_grad
    )


def zeros_like(input, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.zeros(input.shape),
        dtype=dtype,
        requires_grad=requires_grad
    )


def ones_like(input, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.ones(input.shape),
        dtype=dtype,
        requires_grad=requires_grad
    )


def rand(*size, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.random.rand(*size),
        dtype=dtype,
        requires_grad=requires_grad
    )


def randn(*size, dtype=None, requires_grad=False):
    if dtype is None:
        dtype = DEFAULT_DTYPE
    return mytorch.Tensor(
        data=np.random.randn(*size),
        dtype=dtype,
        requires_grad=requires_grad
    )


def arange(start, end, step=1, dtype=None, requires_grad=False):
    return mytorch.Tensor(
        data=np.arange(start, end, step),
        dtype=dtype,
        requires_grad=requires_grad
    )
