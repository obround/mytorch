import numpy as np
import mytorch.autograd as autograd
from mytorch.autograd.graph import *

# TODO: Implement custom types
bool = np.bool_
float16 = np.float16
float32 = np.float32
float64 = np.float64
int16 = np.int16
int32 = np.int32
int64 = np.int64


class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False, grad_fn=None):
        if dtype is None and isinstance(data, (Tensor, np.ndarray)):
            dtype = data.dtype
        self.data = np.asarray(data, dtype=dtype)
        self.dtype = self.data.dtype
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn if self.requires_grad else None
        self.grad = None

        if self.dtype not in [float16, float32, float64]:
            raise ValueError("Only floating point tensors can require gradients")

    def __repr__(self):
        # TODO: What is this pile of trash
        numpy_repr = str(self.numpy()).splitlines()
        numpy_repr = numpy_repr[0] + ("\n" + " " * 7 if len(numpy_repr) > 1 else "") + \
                     ("\n" + " " * 7).join(numpy_repr[1:])
        return f"tensor({numpy_repr}, requires_grad={self.requires_grad})"

    def __hash__(self):
        return hash(id(self))

    def backward(self, gradient=None):
        autograd.backward(self, grad_tensor=gradient)

    def numpy(self):
        return self.data

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    def add(self, other):
        return add(self, other)

    def sub(self, other):
        return sub(self, other)

    def mul(self, other):
        return mul(self, other)

    def div(self, other):
        return div(self, other)

    def neg(self):
        return neg(self)

    def pow(self, exponent):
        return pow(self, exponent)

    def log(self):
        return log(self)

    def sum(self, dim=None, keepdim=False, dtype=None):
        return sum(self, dim, keepdim, dtype)

    def reshape(self, *shape):
        return reshape(self, shape)

    def __rsub__(self, other):
        return ensure_tensor(other) - self

    def __rtruediv__(self, other):
        return ensure_tensor(other) / self

    __add__ = add
    __radd__ = add
    __sub__ = sub
    __mul__ = mul
    __rmul__ = mul
    __truediv__ = div
    __pow__ = pow
    __neg__ = neg


def ensure_tensor(value):
    if isinstance(value, Tensor):
        return value
    return Tensor(value, dtype=float64)


def check_requires_grad(*tensors):
    return any(map(lambda t: t.requires_grad, tensors))


def add(input, other):
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return Tensor(
        data=input.data + other.data,
        requires_grad=check_requires_grad(input, other),
        grad_fn=AddBackward(input, other)
    )


def sub(input, other):
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return Tensor(
        data=input.data - other.data,
        requires_grad=check_requires_grad(input, other),
        grad_fn=SubBackward(input, other)
    )


def mul(input, other):
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return Tensor(
        data=input.data * other.data,
        requires_grad=check_requires_grad(input, other),
        grad_fn=MulBackward(input, other)
    )


def div(input, other):
    input = ensure_tensor(input)
    other = ensure_tensor(other)
    return Tensor(
        data=input.data / other.data,
        requires_grad=check_requires_grad(input, other),
        grad_fn=DivBackward(input, other)
    )


def neg(input):
    input = ensure_tensor(input)
    return Tensor(
        data=-input.data,
        requires_grad=check_requires_grad(input),
        grad_fn=NegBackward(input)
    )


def pow(input, exponent):
    input = ensure_tensor(input)
    exponent = ensure_tensor(exponent)
    return Tensor(
        data=input.data ** exponent.data,
        requires_grad=check_requires_grad(input, exponent),
        grad_fn=PowBackward(input, exponent)
    )


def log(input):
    input = ensure_tensor(input)
    return Tensor(
        data=np.log(input.data),
        requires_grad=check_requires_grad(input),
        grad_fn=LogBackward(input)
    )


def sum(input, dim=None, keepdim=False, dtype=None):
    input = ensure_tensor(input)
    return Tensor(
        data=np.sum(input.data, axis=dim, keepdims=keepdim, dtype=dtype),
        dtype=dtype,
        requires_grad=check_requires_grad(input),
        grad_fn=SumBackward(input)
    )


def reshape(input, shape):
    input = ensure_tensor(input)
    return Tensor(
        data=np.reshape(input.data, shape),
        requires_grad=check_requires_grad(input),
        grad_fn=ReshapeBackward(input, shape=shape)
    )
