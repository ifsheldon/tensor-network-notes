# AUTOGENERATED! DO NOT EDIT! File to edit: ../../2-5.ipynb.

# %% auto 0
__all__ = ['apply_gate', 'pauli_operator', 'rotate']

# %% ../../2-5.ipynb 3
from tensor_network.utils import (
    iterable_have_common,
    inverse_permutation,
    check_quantum_gate,
    check_state_tensor,
    unify_tensor_dtypes,
)
import torch
from typing import List
from einops import einsum

# %% ../../2-5.ipynb 4
def apply_gate(
    *,
    quantum_state: torch.Tensor,
    gate: torch.Tensor,
    target_qubit: int | List[int],
    control_qubit: int | List[int] | None = None,
) -> torch.Tensor:
    """
    Apply a quantum gate to a quantum state tensor. It can also be used to implement controlled gates by specifying control qubits.

    Args:
        quantum_state (torch.Tensor): The quantum state tensor.
        gate (torch.Tensor): The quantum gate tensor.
        target_qubit (int | List[int]): The target qubit(s) to apply the gate to.
        control_qubit (int | List[int] | None): The control qubit(s) for the gate. If None, no control qubits are used.
    Returns:
        torch.Tensor: The new quantum state tensor after applying the gate.
    """
    check_state_tensor(quantum_state)

    # check types
    assert isinstance(target_qubit, (int, list)), "target qubit must be int or list"
    assert control_qubit is None or isinstance(control_qubit, (int, list)), (
        "control_qubit must be int or list"
    )

    # unify types
    if isinstance(target_qubit, int):
        target_qubit = [target_qubit]
    if control_qubit is None:
        control_qubit = []
    elif isinstance(control_qubit, int):
        control_qubit = [control_qubit]
    assert not iterable_have_common(target_qubit, control_qubit), (
        "target qubit and control qubit must not overlap"
    )

    num_qubits = quantum_state.ndim
    num_target_qubit = len(target_qubit)
    check_quantum_gate(gate, num_target_qubit)

    quantum_state, gate = unify_tensor_dtypes(quantum_state, gate)

    # check indices
    for qidx in target_qubit:
        assert 0 <= qidx < num_qubits, f"target qubit index {qidx} out of range"
    for qidx in control_qubit:
        assert 0 <= qidx < num_qubits, f"control qubit index {qidx} out of range"

    if gate.ndim == 2:
        # if in matrix form, reshape to tensor form
        new_shape = [2] * (num_target_qubit * 2)
        gate = gate.reshape(new_shape)

    other_qubits = list(range(num_qubits))
    for qubit_idx in target_qubit:
        other_qubits.remove(qubit_idx)
    for qubit_idx in control_qubit:
        other_qubits.remove(qubit_idx)

    num_other_qubits = len(other_qubits)
    permutation = target_qubit + other_qubits + control_qubit
    state = torch.permute(quantum_state, permutation)
    state_shape = state.shape  # (*target_qubit_shapes, *other_qubit_shapes, *control_qubit_shapes)
    # Flatten the state tensor, so that the shape is (target_qubit_shapes, other_qubit_shapes, -1)
    new_shape = [2] * (num_target_qubit + num_other_qubits) + [-1]
    state = state.reshape(new_shape)
    # only when control qubits are 11111... the gate is applied
    unaffected_state = state[
        ..., :-1
    ]  # (*target_qubit_shapes, *other_qubit_shapes, flattened_dim-1)
    state_to_apply_gate = state[..., -1]  # (*target_qubit_shapes, *other_qubit_shapes)
    # apply gate
    target_qubit_names = [f"t{i}" for i in target_qubit]
    other_qubit_names = [f"o{i}" for i in other_qubits]
    gate_output_qubit_names = [f"g{i}" for i in target_qubit]
    einsum_str = "{gate_dims}, {state_dims} -> {output_dims}".format(
        gate_dims=" ".join(gate_output_qubit_names + target_qubit_names),
        state_dims=" ".join(target_qubit_names + other_qubit_names),
        output_dims=" ".join(gate_output_qubit_names + other_qubit_names),
    )
    new_state = einsum(gate, state_to_apply_gate, einsum_str)
    new_state = new_state.unsqueeze(-1)

    final_state = torch.cat(
        [unaffected_state, new_state], dim=-1
    )  # (*target_qubit_shapes, *other_qubit_shapes, flattened_dim)
    final_state = final_state.reshape(
        state_shape
    )  # (*target_qubit_shapes, *other_qubit_shapes, *control_qubit_shapes)
    inversed_permutation = inverse_permutation(permutation)
    final_state = final_state.permute(inversed_permutation)
    return final_state

# %% ../../3-1.ipynb 3
from typing import Literal
from ..utils import map_float_to_complex


def pauli_operator(
    *,
    pauli: Literal["X", "Y", "Z", "ID"],
    double_precision: bool = False,
    force_complex: bool = False,
) -> torch.Tensor:
    """
    Returns the Pauli operator as a tensor.
    Args:
        pauli (str): The Pauli operator to return. Must be one of 'X', 'Y', 'Z', or 'ID'.
        double_precision (bool): If True, use double precision for the tensor.
        force_complex (bool): If True, force the tensor to be complex.
    Returns:
        torch.Tensor: The Pauli operator as a tensor.
    """
    assert pauli in ["X", "Y", "Z", "ID"], f"Invalid Pauli operator: {pauli}"
    if double_precision:
        dtype_complex = torch.complex128
        if force_complex:
            dtype_default = dtype_complex
        else:
            dtype_default = torch.float64
    else:
        dtype_complex = torch.complex64
        if force_complex:
            dtype_default = dtype_complex
        else:
            dtype_default = torch.float32

    if pauli == "X":
        return torch.tensor([[0, 1], [1, 0]], dtype=dtype_default)
    elif pauli == "Y":
        return torch.tensor([[0, -1j], [1j, 0]], dtype=dtype_complex)
    elif pauli == "Z":
        return torch.tensor([[1, 0], [0, -1]], dtype=dtype_default)
    elif pauli == "ID":
        return torch.eye(2, dtype=dtype_default)
    else:
        raise Exception("Unreachable code")


def _float_convert_to_tensor(
    value: float | torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    elif isinstance(value, float):
        return torch.tensor(value, dtype=dtype, device=device)
    else:
        raise TypeError(f"Expected float or torch.Tensor, got {type(value)}")


def rotate(
    *,
    params_vec: torch.Tensor | None = None,
    ita: torch.Tensor | float | None = None,
    beta: torch.Tensor | float | None = None,
    delta: torch.Tensor | float | None = None,
    gamma: torch.Tensor | float | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Returns the rotation gate as a tensor.
    Args:
        params_vec (torch.Tensor): A 4-element vector containing the parameters [ita, beta, delta, gamma].
        ita (torch.Tensor | float | None): The first parameter of the rotation gate.
        beta (torch.Tensor | float | None): The second parameter of the rotation gate.
        delta (torch.Tensor | float | None): The third parameter of the rotation gate.
        gamma (torch.Tensor | float | None): The fourth parameter of the rotation gate.
        dtype (torch.dtype | None): The data type of the tensor.
        device (torch.device | None): The device to create the tensor on.
    Returns:
        torch.Tensor: The rotation gate as a tensor.
    """

    assert params_vec is not None or (
        ita is not None and beta is not None and delta is not None and gamma is not None
    ), "Either params_vec or ita, beta, delta, and gamma must be provided"
    if params_vec is not None:
        assert isinstance(params_vec, torch.Tensor), "params must be a torch.Tensor"
        assert params_vec.shape == (4,), "params must be a 4-element vector"
        dtype = params_vec.dtype if dtype is None else dtype
        device = params_vec.device if device is None else device
        assert dtype in [torch.float32, torch.float64], "params must be float32 or float64"
        beta, delta, ita, gamma = params_vec[0], params_vec[1], params_vec[2], params_vec[3]
    else:
        ita = _float_convert_to_tensor(ita, device=device, dtype=dtype)
        beta = _float_convert_to_tensor(beta, device=device, dtype=dtype)
        delta = _float_convert_to_tensor(delta, device=device, dtype=dtype)
        gamma = _float_convert_to_tensor(gamma, device=device, dtype=dtype)
        assert ita.shape == beta.shape == delta.shape == gamma.shape == (), (
            "ita, beta, delta, and gamma must be scalars"
        )
        assert ita.dtype == beta.dtype == delta.dtype == gamma.dtype, (
            "ita, beta, delta, and gamma must have the same dtype"
        )
        assert ita.device == beta.device == delta.device == gamma.device, (
            "ita, beta, delta, and gamma must have the same device"
        )
        dtype = ita.dtype if dtype is None else dtype
        device = ita.device if device is None else device
        assert dtype in [torch.float32, torch.float64], (
            "ita, beta, delta, and gamma must be float32 or float64"
        )

    gate_dtype = map_float_to_complex(dtype=dtype)
    # calculate the matrix for the beta terms
    beta_coefficient_matrix = torch.tensor(
        [[-0.5, -0.5], [0.5, 0.5]], device=device, dtype=gate_dtype
    )
    beta_matrix = beta_coefficient_matrix * beta
    # calculate the matrix for the delta terms
    delta_coefficient_matrix = beta_coefficient_matrix.T
    delta_matrix = delta_coefficient_matrix * delta
    # calculate the matrix for the gamma terms
    gamma_2 = gamma / 2
    gamma_coefficient_matrix_cosine = torch.eye(2, device=device, dtype=gate_dtype)
    gamma_coefficient_matrix_sine = torch.tensor([[0, 1], [1, 0]], device=device, dtype=gate_dtype)
    gamma_matrix = gamma_coefficient_matrix_cosine * torch.cos(
        gamma_2
    ) + gamma_coefficient_matrix_sine * torch.sin(gamma_2)
    # set the coefficient matrix in front of e
    coefficient_matrix = torch.tensor([[1, -1], [1, 1]], device=device, dtype=gate_dtype)
    gate = coefficient_matrix * torch.exp(1j * (ita + beta_matrix + delta_matrix)) * gamma_matrix
    return gate