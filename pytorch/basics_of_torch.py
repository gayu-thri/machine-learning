import torch

# =======================================================
print("---- EXAMPLE b = 2*(a^3) ----")
a = torch.tensor(5.0, requires_grad=True)
b = 2 * a**3
# grad of b w.r.t a = 6a^2

b.backward()
# Executes backward pass and computes all backprop gradients
# w.r.t all parameters with requires_grad = True
# and stores in parameter "grad" attribute for every parameter

print("[b = 2 * a ** 3] [a = [5.]]; db/da = ", a.grad)

# =======================================================
print("\n---- EXAMPLE y = wx + b ----")
# Only tensors of floating point can require gradients
x = torch.tensor(5.0)
w = torch.tensor(10.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)

print("X = ", x)
print("W = ", w)
print("B = ", b)

y = w * x + b
print("Y = ", y)

y.backward()
print("\n---CALCULATING GRADIENTS---")
print("dy/dx (have set requires_grad = False) = ", x.grad)
print("dy/dw = ", w.grad)
print("dy/db = ", b.grad)

# =======================================================
print("\n\n---- RANDOM EXPLORATIONS ----\n")

# To calculate gradients of function w.r.t X
x = torch.randn(3, requires_grad=True)
# Whenever we work with tensors, python creates computational graph
y = x + 2
# Each operation => has input and output

# PyTorch automatically creates and stores a function
# which is used in backprop to get gradients
# Gradient function depends on operation
print(f"x: {x}")
"""
tensor([-0.2036,  0.9120,  0.3301], requires_grad=True)
tensor([1.7964, 2.9120, 2.3301], grad_fn=<AddBackward0>)
"""
z = y * y * 2
print(z)
"""
tensor([24.9383, 12.4715, 15.8341], grad_fn=<MulBackward0>)
"""
z = z.mean()
print(z)
"""
tensor(17.7479, grad_fn=<MeanBackward0>)
"""

# Works for scalar,
# else provide vector for jacobian product

z.backward()  # dz/dx
print(x)  # requires_grad=True
print(x.grad)

# doesn't require gradients
x.requires_grad_(False)
print(f"Requires grad False x: {x}")
y = x.detach()
print(f"y = x.detach(): {y}")

with torch.no_grad():
    y = x + 2
    print(f"no grad: {y}")


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights * 3).sum()
    model_output.backward()  # calculates gradients
    print(weights.grad)
    # empty the gradients
    weights.grad.zero_()  # if commented, gradients changes (sums up)
