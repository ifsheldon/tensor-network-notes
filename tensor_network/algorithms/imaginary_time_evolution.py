# AUTOGENERATED! DO NOT EDIT! File to edit: ../../5-1.ipynb.

# %% auto 0
__all__ = ['imaginary_time_evolution']

# %% ../../5-1.ipynb 3
import torch
from typing import List, Tuple
from ..quantum_state.functional import calc_observation
from ..tensor_gates.hamiltonians import heisenberg
from ..tensor_gates.functional import apply_gate
from ..utils.checking import check_state_tensor, check_quantum_gate
from ..utils.mapping import view_gate_matrix_as_tensor, view_gate_tensor_as_matrix

# %% ../../5-1.ipynb 4
def imaginary_time_evolution(
    hamiltonian: torch.Tensor,
    interaction_positions: List[List[int]] | torch.Tensor,
    tau: float,
    iterations: int,
    time_ob: int,
    e0_converge_limit: float,
    tau_min: float,
    *,
    num_qubits: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    init_qubit_state: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform imaginary time evolution on a quantum pure state.

    Args:
        hamiltonian: The Hamiltonian of the system.
        interaction_positions: The positions of the interactions.
        tau: The imaginary time step.
        iterations: The number of iterations.
        time_ob: The number of iterations between each observation.
        e0_converge_limit: The convergence limit of the ground energy.
        tau_min: The convergence limit of the imaginary time step.
        num_qubits: The number of qubits in the system.
        dtype: The dtype of the system.
        device: The device of the system.
        init_qubit_state: The initial state of the system.

    Returns:
        The final state and the ground energy.
    """
    assert iterations > time_ob > 0
    assert e0_converge_limit > 0.0 and tau > tau_min > 0.0
    check_quantum_gate(hamiltonian, assert_tensor_form=True)
    if num_qubits is None:
        assert init_qubit_state is not None, "num_qubits and init_qubit_state cannot be both None"
        dtype = init_qubit_state.dtype
        device = init_qubit_state.device
        assert hamiltonian.device == device, (
            "hamiltonian and init_qubit_state must be on the same device"
        )
        quantum_state = init_qubit_state / init_qubit_state.norm()
    elif init_qubit_state is None:
        assert num_qubits is not None and dtype is not None and device is not None, (
            "num_qubits, dtype and device must be provided if init_qubit_state is not provided"
        )
        init_qubit_state = torch.randn(*([2] * num_qubits), dtype=dtype, device=device)
        quantum_state = init_qubit_state / init_qubit_state.norm()
    else:
        raise ValueError("num_qubits and init_qubit_state cannot be both provided")

    assert isinstance(interaction_positions, (List, torch.Tensor)), (
        "interaction_positions must be a List or torch.Tensor"
    )
    if isinstance(interaction_positions, List):
        interaction_positions = torch.tensor(interaction_positions, dtype=torch.long, device=device)

    imag_time_evo_op = view_gate_matrix_as_tensor(
        torch.matrix_exp(-tau * view_gate_tensor_as_matrix(hamiltonian))
    )

    e0 = 1.0
    inversed_temperature = 0.0

    for t in range(iterations):
        for positions in interaction_positions:
            quantum_state = apply_gate(
                quantum_state=quantum_state, gate=imag_time_evo_op, target_qubit=positions.tolist()
            )

        quantum_state = quantum_state / quantum_state.norm()
        inversed_temperature += tau

        if t % time_ob == 0:
            ground_energy = 0.0
            for positions in interaction_positions:
                ground_energy += calc_observation(
                    quantum_state, hamiltonian, qubit_idx=positions.tolist()
                )

            print(f"\nAt iteration {t}")
            print(
                f"  Inversed Temperature = {inversed_temperature}, Ground Energy={ground_energy.item()}"
            )
            if abs(ground_energy - e0) < e0_converge_limit * tau:
                tau *= 0.5
                imag_time_evo_op = torch.matrix_exp(-tau * hamiltonian.reshape(4, 4)).reshape(
                    2, 2, 2, 2
                )
                print(f"  Tau is reduced to {tau} since the ground energy is converged")
            if tau < tau_min:
                print(f"  Tau is less than {tau_min}, terminating the imaginary time evolution")
                break
            e0 = ground_energy

    return quantum_state, e0
