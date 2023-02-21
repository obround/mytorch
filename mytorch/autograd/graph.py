import mytorch


# Function from HIPS/autograd
def unbroadcast(target, grad, broadcast_idx=0):
    while grad.ndim > target.ndim:
        grad = grad.sum(dim=broadcast_idx)
    for axis, size in enumerate(target.shape):
        if size == 1:
            grad = grad.sum(dim=axis, keepdim=True)
    return grad


# TODO: Make iterative
def toposort(grad_node, tensor):
    topo = []
    visited = set()

    def worker(n, t):
        if n not in visited:
            visited.add(n)
            for child_tensor, param in zip(n.deps, n.next_functions):
                worker(param, child_tensor)
            topo.append((n, t))

    worker(grad_node, tensor)
    return topo


class GradNode:
    def __init__(self, deps):
        self.deps = deps
        self.next_functions = tuple(
            parent.grad_fn if parent.grad_fn else AccumulateGrad()
            for parent in self.deps
        )

    def __call__(self, grad):
        raise NotImplementedError()


class AccumulateGrad(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return ()


class AddBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return (unbroadcast(self.deps[0], grad),
                unbroadcast(self.deps[1], grad))


class SubBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return (unbroadcast(self.deps[0], grad),
                unbroadcast(self.deps[1], -grad))


class MulBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return (unbroadcast(self.deps[0], grad * self.deps[1]),
                unbroadcast(self.deps[1], grad * self.deps[0]))


class DivBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return (unbroadcast(self.deps[0], grad / self.deps[1]),
                unbroadcast(self.deps[1], -grad * self.deps[0] / self.deps[1] ** 2))


class NegBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return -grad,


class PowBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return (unbroadcast(self.deps[0], grad * self.deps[1] * self.deps[0] ** (self.deps[1] - 1)),
                unbroadcast(self.deps[1], grad * self.deps[0].log() * self.deps[0] ** self.deps[1]))


class LogBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        return grad / self.deps[0],


class SumBackward(GradNode):
    def __init__(self, *args):
        super().__init__(args)

    def __call__(self, grad):
        # NOTE: Changed to 'grad * ...'
        return grad * mytorch.ones_like(self.deps[0]),


class ReshapeBackward(GradNode):
    def __init__(self, *args, shape):
        self.shape = shape
        super().__init__(args)

    def __call__(self, grad):
        return mytorch.reshape(grad, self.shape),
