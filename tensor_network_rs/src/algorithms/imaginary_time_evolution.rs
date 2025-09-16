use crate::quantum_state::functional::calc_observation;
use crate::tensor_gates::functional::apply_gate;
use crate::utils::mapping::{view_gate_matrix_as_tensor, view_gate_tensor_as_matrix};
use tch::{Device, Kind, Tensor};

#[allow(clippy::too_many_arguments)]
/// Imaginary Time Evolution (ITE) for a 2-body Hamiltonian acting on given
/// `interaction_positions` (pairs of qubit indices).
///
/// Mirrors the Python helper: evolves by repeatedly applying `exp(-tau H)`
/// in tensor form, normalizing the state, and adaptively halving `tau`
/// when the ground energy stabilizes below `e0_converge_limit * tau`.
pub fn imaginary_time_evolution(
    hamiltonian: &Tensor,
    interaction_positions: Vec<Vec<i64>>,
    mut tau: f64,
    iterations: i64,
    time_ob: i64,
    e0_converge_limit: f64,
    tau_min: f64,
    num_qubits: Option<i64>,
    dtype: Option<Kind>,
    device: Option<Device>,
    init_qubit_state: Option<Tensor>,
) -> (Tensor, Tensor) {
    assert!(iterations > time_ob && time_ob > 0);
    assert!(e0_converge_limit > 0.0 && tau > tau_min && tau_min > 0.0);
    // Initialize state
    let (mut state, _k, _dev) = if let Some(nq) = num_qubits {
        let k = dtype.expect("dtype required when num_qubits provided");
        let dev = device.expect("device required when num_qubits provided");
        let s = if let Some(s0) = init_qubit_state {
            s0.shallow_clone()
        } else {
            Tensor::randn(vec![2_i64; nq as usize], (k, dev))
        };
        let s_normed = &s / s.norm();
        (s_normed, k, dev)
    } else {
        let s0 = init_qubit_state.expect("either num_qubits or init_qubit_state must be provided");
        let k = s0.kind();
        let dev = s0.device();
        (s0.shallow_clone() / s0.norm(), k, dev)
    };

    // Evolution operator in tensor form
    let mut evo = {
        let mat = view_gate_tensor_as_matrix(hamiltonian, None);
        let op = (-tau) * mat;
        let mexp = op.matrix_exp();
        view_gate_matrix_as_tensor(&mexp, Some(2))
    };

    let mut e0 = Tensor::from(1.0);
    let mut _inv_temp = 0.0;
    for t in 0..iterations {
        for pos in &interaction_positions {
            state = apply_gate(&state, &evo, pos.clone(), None);
        }
        state = &state / state.norm();
        _inv_temp += tau;

        if t % time_ob == 0 {
            let mut ground = Tensor::from(0.0);
            for pos in &interaction_positions {
                ground = &ground + calc_observation(&state, hamiltonian, pos.clone(), true);
            }
            let diff = (&ground - &e0).abs().double_value(&[]);
            if diff < e0_converge_limit * tau {
                tau *= 0.5;
                let mat = view_gate_tensor_as_matrix(hamiltonian, None);
                let op = (-tau) * mat;
                let mexp = op.matrix_exp();
                evo = view_gate_matrix_as_tensor(&mexp, Some(2));
            }
            if tau < tau_min {
                break;
            }
            e0 = ground;
        }
    }
    (state, e0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_gates::functional::heisenberg;

    #[test]
    #[ignore]
    fn test_ite_runs() {
        let h = heisenberg(1.0, 1.0, 1.0, true, false);
        let (s, e0) = imaginary_time_evolution(
            &h,
            vec![vec![0, 1]],
            0.1,
            10,
            2,
            1e-6,
            1e-3,
            Some(2),
            Some(Kind::ComplexDouble),
            Some(Device::Cpu),
            None,
        );
        assert_eq!(s.size(), vec![2, 2]);
        assert!(e0.double_value(&[]).is_finite());
    }
}
