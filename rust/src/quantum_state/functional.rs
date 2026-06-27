//! Functional helpers for dense quantum states.

use tch::{Kind, Tensor};

use crate::utils::checking::{check_quantum_gate, check_state_tensor};
use crate::utils::devices::linalg_work_device;

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

/// Calculate bond energies of a dense quantum state.
pub fn observe_bond_energies(
    state: &Tensor,
    hamiltonians: &[Tensor],
    interaction_positions: &[Vec<i64>],
) -> Tensor {
    check_state_tensor(state);
    assert!(
        !hamiltonians.is_empty(),
        "at least one Hamiltonian is required"
    );
    assert!(
        !interaction_positions.is_empty(),
        "interaction_positions must be non-empty"
    );
    let hs = if hamiltonians.len() == 1 {
        (0..interaction_positions.len())
            .map(|_| hamiltonians[0].shallow_clone())
            .collect::<Vec<_>>()
    } else {
        assert_eq!(
            hamiltonians.len(),
            interaction_positions.len(),
            "hamiltonian and interaction_positions must have the same length"
        );
        hamiltonians
            .iter()
            .map(Tensor::shallow_clone)
            .collect::<Vec<_>>()
    };
    for hamiltonian in &hs {
        check_quantum_gate(hamiltonian, None, true);
    }
    let energies = hs
        .iter()
        .zip(interaction_positions.iter())
        .map(|(hamiltonian, position)| {
            calc_observation(
                state,
                &hamiltonian.to_device(state.device()),
                position,
                false,
            )
        })
        .collect::<Vec<_>>();
    Tensor::stack(&energies, 0)
}

/// Calculate bipartite entanglement entropy at one or more split positions.
pub fn bipartite_entanglement_entropy(
    state: &Tensor,
    qubit_idx: Option<&[i64]>,
    eps: f64,
) -> Tensor {
    check_state_tensor(state);
    assert!(eps > 0.0, "eps must be positive");
    let num_qubits = state.dim() as i64;
    let owned;
    let indices = match qubit_idx {
        Some(indices) => {
            assert!(!indices.is_empty(), "qubit_idx must be non-empty");
            indices
        }
        None => {
            owned = (1..num_qubits).collect::<Vec<_>>();
            &owned
        }
    };
    let shape = state.size();
    let entropies = indices
        .iter()
        .map(|&idx| {
            assert!(
                1 <= idx && idx < num_qubits,
                "qubit_idx must be in the range [1, num_qubits)"
            );
            let left_dim = shape[..idx as usize].iter().product::<i64>();
            let mat = state.reshape([left_dim, -1]);
            let work_device = linalg_work_device(mat.device());
            let singular_values =
                Tensor::linalg_svdvals(&mat.to_device(work_device), "").to_device(state.device());
            let eigenvalues = singular_values.pow_tensor_scalar(2.0);
            -eigenvalues.inner(&(eigenvalues.shallow_clone() + eps).log())
        })
        .collect::<Vec<_>>();
    Tensor::stack(&entropies, 0)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;
    use crate::tensor_gates::functional::{Pauli, gate_outer_product, pauli_operator};

    #[test]
    fn bipartite_entropy_of_product_state_is_zero() {
        let state = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]).reshape([2, 2]);
        let entropy = bipartite_entanglement_entropy(&state, None, 1e-14);
        assert!(entropy.allclose(
            &Tensor::zeros([1], (Kind::Float, Device::Cpu)),
            1e-6,
            1e-8,
            false
        ));
    }

    #[test]
    fn observe_bond_energies_returns_one_value_per_position() {
        let state = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]).reshape([2, 2]);
        let z = pauli_operator(Pauli::Z, false, false, Device::Cpu);
        let zz = gate_outer_product(&[z.shallow_clone(), z], false);
        let energies = observe_bond_energies(&state, &[zz], &[vec![0, 1]]);
        assert_eq!(energies.size(), vec![1]);
        assert!((energies.double_value(&[0]) - 1.0).abs() < 1e-6);
    }
}
