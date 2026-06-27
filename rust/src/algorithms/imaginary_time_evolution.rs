//! Dense imaginary-time evolution.

use tch::{Device, Kind, Tensor};

use crate::quantum_state::calc_observation;
use crate::tensor_gates::functional::apply_gate;
use crate::utils::checking::check_quantum_gate;
use crate::utils::mapping::{view_gate_matrix_as_tensor, view_gate_tensor_as_matrix};

/// Initial state selection for imaginary-time evolution.
pub enum InitialState<'a> {
    /// Use an existing state tensor.
    Tensor(&'a Tensor),
    /// Generate a random state.
    Random {
        /// Number of qubits in the generated state.
        num_qubits: i64,
        /// Tensor dtype for the generated state.
        kind: Kind,
        /// Device for the generated state.
        device: Device,
    },
}

/// Perform imaginary-time evolution on a dense quantum state.
pub fn imaginary_time_evolution(
    hamiltonian: &Tensor,
    interaction_positions: &[Vec<i64>],
    mut tau: f64,
    iterations: i64,
    time_ob: i64,
    e0_converge_limit: f64,
    tau_min: f64,
    init: InitialState<'_>,
) -> (Tensor, Tensor) {
    assert!(iterations > time_ob && time_ob > 0);
    assert!(e0_converge_limit > 0.0 && tau > tau_min && tau_min > 0.0);
    check_quantum_gate(hamiltonian, None, true);
    let mut state = match init {
        InitialState::Tensor(tensor) => tensor / tensor.norm(),
        InitialState::Random {
            num_qubits,
            kind,
            device,
        } => {
            let random = Tensor::randn(vec![2; num_qubits as usize].as_slice(), (kind, device));
            &random / random.norm()
        }
    };
    let hamiltonian = hamiltonian.to_device(state.device());
    let mut op = view_gate_matrix_as_tensor(
        &((-tau * view_gate_tensor_as_matrix(&hamiltonian, None)).matrix_exp()),
        None,
    );
    let mut e0 = Tensor::from(1.0)
        .to_kind(state.kind())
        .to_device(state.device());
    for t in 0..iterations {
        for positions in interaction_positions {
            state = apply_gate(&state, &op, positions, &[]);
        }
        state = &state / state.norm();
        if t % time_ob == 0 {
            let mut ground_energy = Tensor::zeros([], (state.kind(), state.device()));
            for positions in interaction_positions {
                ground_energy += calc_observation(&state, &hamiltonian, positions, true);
            }
            if (&ground_energy - &e0).abs().double_value(&[]) < e0_converge_limit * tau {
                tau *= 0.5;
                op = view_gate_matrix_as_tensor(
                    &((-tau * view_gate_tensor_as_matrix(&hamiltonian, None)).matrix_exp()),
                    None,
                );
            }
            if tau < tau_min {
                break;
            }
            e0 = ground_energy;
        }
    }
    (state, e0)
}
