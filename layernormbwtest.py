# %%
# %%
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)
import time
import numpy as np

# %%
# %%
rows = 10
hidden_dim = 8

def rand(shape):
    return np.random.rand(*shape)

inp = rand((rows, hidden_dim))
out_grad = rand((rows,hidden_dim))
gamma = rand((rows,1))
beta = rand((rows,1))


def custom():
    inp_mt = minitorch.tensor_from_numpy(inp, backend=backend, requires_grad=True)
    gamma_mt = minitorch.tensor_from_numpy(gamma, backend=backend, requires_grad=True)
    beta_mt = minitorch.tensor_from_numpy(beta, backend=backend, requires_grad=True)
    out_grad_mt = minitorch.tensor_from_numpy(out_grad, backend=backend, requires_grad=True)

    out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
    out_mt.backward(out_grad_mt)

    return inp_mt.grad, gamma_mt.grad, beta_mt.grad

def baseline():
    f_input = minitorch.tensor_from_numpy(inp, backend=backend, requires_grad=True)
    f_gamma = minitorch.tensor_from_numpy(gamma, backend=backend, requires_grad=True)
    f_out_grad = minitorch.tensor_from_numpy(out_grad, backend=backend, requires_grad=True)

    f_means = f_input.mean(dim=1)
    f_vars = f_input.var(dim=1)
    f_stds = minitorch.tensor(np.sqrt(f_vars.to_numpy()).reshape(-1, 1).tolist(), backend=backend, requires_grad=True)

    xhat = (f_input - f_means) / f_stds
    dxhat = f_out_grad * f_gamma
    f_betta_grad = f_out_grad.sum(dim=0)
    f_gamma_grad = (f_out_grad * xhat).sum(dim=0)
    dinp = dxhat.sum(dim=1) + xhat * (dxhat * xhat).sum(dim=1)
    dinp = dxhat - dinp / hidden_dim
    dinp = dinp / f_stds
    return dinp, f_gamma_grad, f_betta_grad




#     return res


# %%


# %%
inp_grad_mt, gamma_mt, beta_mt = custom()
dinp, f_gamma_grad, f_betta_grad = baseline()
# print(c-b)


# %%
# beta_mt
print(f_gamma_grad)
print(f_betta_grad)

# %%
# f_betta_grad

# %%



