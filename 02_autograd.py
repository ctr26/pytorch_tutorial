# %%
import torch
import numpy as np
# Tensor factory methods have a ``requires_grad`` flag
x = torch.tensor([1., 2., 3], requires_grad=True)

# With requires_grad=True, you can still do all the operations you previously
# could
y = torch.tensor([4., 5., 6], requires_grad=True)
z = x + y
print(z)

# BUT z knows something extra.
print(z.grad_fn)

# Let's sum up all the entries in z
s = z.sum()
print(s)
print(s.grad_fn)

# calling .backward() on any variable will run backprop, starting from it.
s.backward()
print(x.grad)

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = x.to('cuda')

# %% Numpy bridge
# Tensor to NumPy array

x_np = x.detach().numpy()
# NumPy array to Tensor

n = np.ones(5)
t = torch.from_numpy(n)
# %%
