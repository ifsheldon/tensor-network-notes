//! Dense ground-state helper.

use tch::{IndexOp, Tensor};

use crate::tensor_gates::functional::apply_gate;
use crate::utils::checking::check_quantum_gate;

/// Calculate the lowest eigenstates of a small dense Hamiltonian.
pub fn calc_ground_state(
    hamiltonians: &[Tensor],
    interact_positions: &[Vec<i64>],
    num_qubits: i64,
    smallest_k: i64,
) -> (Tensor, Tensor) {
    assert!(smallest_k >= 1);
    assert!(num_qubits >= 2);
    assert!(!hamiltonians.is_empty());
    let gate_qubits = check_quantum_gate(&hamiltonians[0], None, false);
    assert!(gate_qubits <= num_qubits);
    let hs: Vec<&Tensor> = if hamiltonians.len() == 1 {
        (0..interact_positions.len())
            .map(|_| &hamiltonians[0])
            .collect()
    } else {
        assert_eq!(hamiltonians.len(), interact_positions.len());
        hamiltonians.iter().collect()
    };
    let dim = 2_i64.pow(num_qubits as u32);
    let device = hamiltonians[0].device();
    let kind = hamiltonians[0].kind();
    let mut columns = Vec::new();
    for basis in 0..dim {
        let mut data = vec![0.0_f32; dim as usize];
        data[basis as usize] = 1.0;
        let mut state = Tensor::from_slice(&data)
            .to_kind(kind)
            .to_device(device)
            .reshape(vec![2; num_qubits as usize].as_slice());
        let mut accum = Tensor::zeros_like(&state);
        for (h, pos) in hs.iter().zip(interact_positions.iter()) {
            accum += apply_gate(&state, h, pos, &[]);
        }
        state = accum.reshape([dim]);
        columns.push(state);
    }
    let dense = Tensor::stack(&columns, 1);
    let (eigvals, eigvecs) = Tensor::internal_linalg_eigh(&dense, "L", true);
    let energies = eigvals.i(..smallest_k);
    let states = eigvecs.i((.., ..smallest_k)).transpose(0, 1);
    if smallest_k == 1 {
        (states.squeeze_dim(0), energies)
    } else {
        (states, energies)
    }
}
