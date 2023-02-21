import mytorch as torch

# Scalar tensors
print("-" * 65)
print("Example 1: Scalar tensors")
print("-" * 65)

a = torch.tensor(3., dtype=torch.float32, requires_grad=True)
b = torch.tensor(10., dtype=torch.float32, requires_grad=True)
c = 2 + (a + b ** 2) / (a + b + a * b)

print("a =", a)
print("b =", b)
print("c = 2 + (a + b ** 2) / (a + b + a * b)")
print("  =", c)

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

# Non-scalar tensors with broadcasting
print("-" * 65)
print("Example 2: Non-scalar tensors (with broadcasting)")
print("-" * 65)
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
