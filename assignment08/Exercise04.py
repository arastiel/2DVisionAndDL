import torch
import math

# Resources:
# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

# Test Run Output:
# Result: y = 1.1398130655288696 + -1.033353136392634e-08 x + -0.6496093273162842 x^2 + 3.1947650080965673e-10 x^3 +
# 0.07613455504179001 x^4 + -8.122776756769312e-11 x^5 + -0.0033741670195013285 x^6

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")  # Uncomment this to run on GPU

# Create Tensors to hold input and outputs.
# By default, requires_grad=False, which indicates that we do not need to
# compute gradients with respect to these Tensors during the backward pass.
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.cos(x)

# Create random Tensors for weights. For a sixth order polynomial, we need
# 7 weights: y = a + b x + c x^2 + d x^3 + e x^4 + f x^5 + g x^6
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Tensors during the backward pass.
a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)
e = torch.randn((), device=device, dtype=dtype, requires_grad=True)
f = torch.randn((), device=device, dtype=dtype, requires_grad=True)
g = torch.randn((), device=device, dtype=dtype, requires_grad=True)

tensors = [a, b, c, d, e, f, g]

learning_rate = 1e-4
optimizer = torch.optim.Adam(tensors, lr=learning_rate)
for t in range(55000):
    # Forward pass: compute predicted y using operations on Tensors.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3 + e * x ** 4 + f * x ** 5 + g * x ** 6

    # Compute and print loss using operations on Tensors.
    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    if t % 1000 == 999:
        print(t, loss.item())
        # if loss smaller than 10 we are satisfied
        if loss.item() < 10:
            break

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

print(
    f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3 + {e.item()} x^4 + {f.item()} x^5 + {g.item()} x^6')
# 2p