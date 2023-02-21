# mytorch
Easily extensible autograd implemented python with pytorch API.

## Examples
`mytorch` supports the computation of arbitrarily high derivatives for both scalars and non-scalars. Both `torch.autograd.backward` and `torch.autograd.grad` are supported.
```python
import mytorch as torch

a = torch.tensor(3., dtype=torch.float32, requires_grad=True)
b = torch.tensor(10., dtype=torch.float32, requires_grad=True)
c = 2 + (a + b ** 2) / (a + b + a * b)

print("a =", a)
print("b =", b)
print("c = 2 + (a + b ** 2) / (a + b + a * b) =", c)

# NOTE: You could also use c.backward() to accumulate the gradients in a.grad and b.grad
dc_da, dc_db = torch.autograd.grad(c, [a, b])
# NOTE: To get higher order derivatives like below, pytorch would require ∂c/∂a and
# ∂c/∂b to be calculated with create_graph=True; mytorch does not require it
d2c_da2 = torch.autograd.grad(dc_da, [a])[0]
d2c_db2 = torch.autograd.grad(dc_db, [b])[0]
print(f"∂c/∂a = {dc_da}")
print(f"∂c/∂b = {dc_db}")
print(f"∂²c/∂a² = {d2c_da2}")
print(f"∂²c/∂b² = {d2c_db2}")
```
Output:
```python
a = tensor(3.0, requires_grad=True)
b = tensor(10.0, requires_grad=True)
c = 2 + (a + b ** 2) / (a + b + a * b)
  = tensor(4.395348787307739, requires_grad=True)
∂c/∂a = tensor(-0.5895078420767982, requires_grad=True)
∂c/∂b = tensor(0.24229313142239048, requires_grad=True)
∂²c/∂a² = tensor(0.3016086633881293, requires_grad=True)
∂²c/∂b² = tensor(0.0014338360144389717, requires_grad=True)
```

Here is a non-scalar example (with broadcasting):
```python
import mytorch as torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
b = torch.tensor([7, 8, 9], dtype=torch.float32, requires_grad=True)
# b is broadcasted
c = a + b

print("a =", a)
print("b =", b)
print("c =", c)
c.backward(torch.ones(2, 3))
print("∂c/∂a =", a.grad)
print("∂c/∂b =", b.grad)
```
Output:
```python
a = tensor([[1. 2. 3.]
            [4. 5. 6.]], requires_grad=True)
b = tensor([7. 8. 9.], requires_grad=True)
c = tensor([[ 8. 10. 12.]
            [11. 13. 15.]], requires_grad=True)
∂c/∂a = tensor([[1. 1. 1.]
                [1. 1. 1.]], requires_grad=False)
∂c/∂b = tensor([2. 2. 2.], requires_grad=False)
```
