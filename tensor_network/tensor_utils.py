# AUTOGENERATED! DO NOT EDIT! File to edit: ../1-4.ipynb.

# %% auto 0
__all__ = ['identity_tensor', 'zeros_state']

# %% ../1-4.ipynb 0
import torch

# %% ../1-4.ipynb 5
def identity_tensor(order: int, dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    dims = [dim] * order
    I = torch.zeros(*dims, dtype=dtype)
    for i in range(dim):
        indices = [i] * order
        I[tuple(indices)] = 1.

    return I

# %% ../3-1.ipynb 7
def zeros_state(*, num_qubits: int, dtype: torch.dtype) -> torch.Tensor:
    assert num_qubits > 0, "num_qubits must be positive"
    assert dtype in [torch.complex64, torch.complex128], "dtype must be complex64 or complex128"
    state = torch.zeros((2 ** num_qubits, ), dtype=dtype)
    state[0] = 1.0
    shape = [2] * num_qubits
    state = state.reshape(shape)
    return state
