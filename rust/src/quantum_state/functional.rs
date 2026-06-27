//! Functional helpers for dense quantum states.

use tch::{Kind, Tensor};

use crate::utils::checking::{check_quantum_gate, check_state_tensor};

/// Calculate the reduced density matrix for selected qubits.
pub fn calc_reduced_density_matrix(state: &Tensor, qubit_idx: &[i64]) -> Tensor {
    assert!(!qubit_idx.is_empty(), "qubit_idx must be non-empty");
    let num_qubits = state.dim() as i64;
    for &idx in qubit_idx {
        assert!(
            0 <= idx && idx < num_qubits,
            "qubit_idx must be in [0, num_qubits - 1]"
        );
    }
    let all: Vec<i64> = (0..num_qubits).collect();
    let dims_to_reduce: Vec<i64> = all
        .iter()
        .copied()
        .filter(|idx| !qubit_idx.contains(idx))
        .collect();
    let permutation: Vec<i64> = qubit_idx
        .iter()
        .copied()
        .chain(dims_to_reduce.iter().copied())
        .collect();
    let state = state.permute(permutation);
    let shape = state.size();
    let keep = qubit_idx
        .iter()
        .map(|&idx| shape[idx as usize])
        .product::<i64>();
    let reduce = shape.iter().product::<i64>() / keep;
    let state = state.reshape([keep, reduce]);
    state.matmul(&state.conj().transpose(0, 1))
}

/// Calculate the expectation value of an operator on a quantum state.
pub fn calc_observation(
    state: &Tensor,
    operator: &Tensor,
    qubit_idx: &[i64],
    fast_mode: bool,
) -> Tensor {
    let rdm = calc_reduced_density_matrix(state, qubit_idx);
    let num_qubits_operator = check_quantum_gate(operator, None, false);
    assert_eq!(
        num_qubits_operator,
        qubit_idx.len() as i64,
        "The number of qubits of the operator does not match the number of qubits of the state"
    );
    let operator = operator.to_device(state.device());
    let operator_mat = operator.reshape([
        2_i64.pow(num_qubits_operator as u32),
        2_i64.pow(num_qubits_operator as u32),
    ]);
    if fast_mode {
        (rdm * operator_mat.transpose(0, 1)).sum(None::<Kind>)
    } else {
        rdm.matmul(&operator_mat).trace()
    }
}

/// Calculate onsite entanglement entropy for selected qubits.
pub fn calc_onsite_entanglement_entropy(
    state: &Tensor,
    qubit_idx: Option<&[i64]>,
    eps: f64,
) -> Tensor {
    check_state_tensor(state);
    let n_qubits = state.dim() as i64;
    let owned;
    let qubit_idx = match qubit_idx {
        Some(indices) => {
            assert!(!indices.is_empty(), "qubit_idx must be a non-empty list");
            indices
        }
        None => {
            owned = (0..n_qubits).collect::<Vec<_>>();
            &owned
        }
    };
    let entropies: Vec<Tensor> = qubit_idx
        .iter()
        .map(|&idx| {
            assert!(
                idx < n_qubits,
                "qubit_idx must be less than the number of qubits"
            );
            assert!(idx >= 0, "qubit_idx must be non-negative");
            let rdm = calc_reduced_density_matrix(state, &[idx]);
            let eigvals = rdm.linalg_eigvalsh("L");
            -eigvals.inner(&(eigvals.shallow_clone() + eps).log())
        })
        .collect();
    Tensor::stack(&entropies, 0)
}

/// Project a quantum state onto a one-qubit state.
pub fn project_state(
    state: &Tensor,
    project_qubit_state: &Tensor,
    project_qubit_idx: i64,
) -> Tensor {
    check_state_tensor(state);
    assert!(
        project_qubit_state.dim() == 1 && project_qubit_state.size()[0] == 2,
        "project_qubit_state must be a 1D tensor with 2 elements"
    );
    let project_qubit_state = project_qubit_state
        .to_device(state.device())
        .to_kind(state.kind());
    let new_state = state.tensordot(&project_qubit_state, [project_qubit_idx], [0]);
    &new_state / new_state.norm()
}
