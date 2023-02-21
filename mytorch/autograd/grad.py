from collections import defaultdict

import mytorch
from mytorch.functions import tensor
from mytorch.autograd.graph import toposort


def grad(output, inputs, grad_output=None, allow_unused=False):
    if not output.requires_grad:
        raise ValueError("output cannot be a tensor with requires_grad=False")
    if output.shape != () and grad_output is None:
        raise ValueError("The JVP vector must be specified for non-scalar outputs")
    if not all(inp.requires_grad for inp in inputs):
        raise ValueError("One or more the differentiated tensors has requires_grad=False")

    grads = defaultdict(lambda: tensor(0.))
    grads[output] = tensor(1.) if grad_output is None else mytorch.ensure_tensor(grad_output)
    if output.grad_fn:  # If grad(x, [x]) where x is a leaf (and x.grad_fn == None)
        for node, current_tensor in reversed(toposort(output.grad_fn, output)):
            for child, grad in zip(node.deps, node(grads[current_tensor])):
                if child.requires_grad:
                    grads[child] += grad
    result = [None] * len(inputs)
    for i, inp in enumerate(inputs):
        if not allow_unused and inp not in grads:
            raise RuntimeError("One or more of the differentiated tensors appears to not have been "
                               "used in the graph. Set allow_unused=True if this is desired")
        result[i] = grads[inp]
    return tuple(result)


def backward(tensor_, grad_tensor=None, inputs=None):
    if not tensor_.requires_grad:
        raise ValueError("output cannot be a tensor with requires_grad=False")
    if tensor_.shape != () and grad_tensor is None:
        raise ValueError("The JVP vector must be specified for non-scalar outputs")
    if grad_tensor and grad_tensor.shape != tensor_.shape:
        raise ValueError(f"Mismatch in shape of grad_tensor:{grad_tensor.shape} and tensor_:{tensor_.shape}")

    # We only temporarily change tensor_.grad to make gradient calculation easier
    prev_grad = tensor_.grad
    tensor_.grad = tensor(1.) if grad_tensor is None else mytorch.ensure_tensor(grad_tensor)
    if tensor_.grad_fn:  # If backward(x, inputs=[x]) where x is a leaf (and x.grad_fn == None)
        for node, current_tensor in reversed(toposort(tensor_.grad_fn, tensor_)):
            for child, grad in zip(node.deps, node(current_tensor.grad)):
                if child.requires_grad and (child in inputs if inputs is not None else True):
                    if child.grad is None:
                        child.grad = tensor(0.)
                    child.grad += grad
    tensor_.grad = prev_grad
