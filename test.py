# %%
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)
import time
import numpy as np

# %%
batch_size = 1
nhead = 8
from_len = 10
to_len = 20

# %%
def rand(shape):
    return np.random.rand(*shape)

out_grad = rand((batch_size, nhead, from_len, to_len))
inp = rand((batch_size, nhead, from_len, to_len))


def custom():
    out_grad_mt = minitorch.tensor_from_numpy(out_grad, backend=backend, requires_grad=True)
    inp_mt = minitorch.tensor_from_numpy(inp, backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor_from_numpy(np.zeros((batch_size, 1, 1, to_len)), backend=backend, requires_grad=True)
    soft_inp_mt = inp_mt.attn_softmax(mask_mt)

    start_time = time.time()
    soft_inp_mt.backward(out_grad_mt)
    end_time = time.time()

    return inp_mt.grad

def baseline():
    out_grad_mt = minitorch.tensor_from_numpy(out_grad, backend=backend, requires_grad=True)
    inp_mt = minitorch.tensor_from_numpy(inp, backend=backend, requires_grad=True)
    soft_inp_mt = minitorch.nn.softmax(inp_mt, dim=3)

    start_time = time.time()
    tsum = out_grad_mt * soft_inp_mt
    tsum = tsum.sum(dim=3).view(tsum.shape[0], tsum.shape[1], tsum.shape[2], 1)
    res = soft_inp_mt * (out_grad_mt - tsum)
    end_time = time.time()

    return res


# %%


# %%
c = custom()
b = baseline()
print(c-b)

# %%
out_grad

# %%
out_grad.shape


# %%



