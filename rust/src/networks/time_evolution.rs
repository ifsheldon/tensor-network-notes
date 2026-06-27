//! Trainable time-evolution circuits.

use tch::{Kind, Tensor, nn};

use crate::networks::adqc::ADQCNet;
use crate::tensor_gates::functional::{Pauli, apply_gate, pauli_operator};
use crate::types::GatePattern;
use crate::utils::mapping::view_gate_matrix_as_tensor;

/// Magnetic-field directions for a polarization gate.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SpinDirection {
    /// X direction.
    X,
    /// Y direction.
    Y,
    /// Z direction.
    Z,
}

/// A gate applying a trainable local magnetic field.
#[derive(Debug)]
pub struct PolarizationGate {
    params: Vec<(SpinDirection, Tensor)>,
    target_qubit: i64,
    time_slice: f64,
}

impl PolarizationGate {
    /// Construct a polarization gate.
    pub fn new(
        vs: &nn::Path<'_>,
        time_slice: f64,
        target_qubit: i64,
        directions: &[SpinDirection],
    ) -> Self {
        assert!(!directions.is_empty() && directions.len() <= 3);
        assert!(time_slice > 0.0);
        assert!(target_qubit >= 0);
        let params = directions
            .iter()
            .copied()
            .map(|direction| {
                let name = match direction {
                    SpinDirection::X => "x",
                    SpinDirection::Y => "y",
                    SpinDirection::Z => "z",
                };
                (direction, vs.randn(name, &[1], 0.0, 1.0))
            })
            .collect();
        Self {
            params,
            target_qubit,
            time_slice,
        }
    }

    /// Forward pass.
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let device = tensor.device();
        let mut spin = Tensor::zeros([2, 2], (Kind::ComplexFloat, device));
        for (direction, param) in &self.params {
            let pauli = match direction {
                SpinDirection::X => Pauli::X,
                SpinDirection::Y => Pauli::Y,
                SpinDirection::Z => Pauli::Z,
            };
            spin += param.to_device(device) * (pauli_operator(pauli, false, true, device) / 2.0);
        }
        let imag = Tensor::complex(
            &Tensor::zeros([], (Kind::Float, device)),
            &Tensor::ones([], (Kind::Float, device)),
        );
        let gate = (imag * (-self.time_slice) * spin).matrix_exp();
        apply_gate(tensor, &gate, &[self.target_qubit], &[])
    }
}

/// ADQC network with fixed coupling gates and trainable polarization gates.
#[derive(Debug)]
pub struct ADQCTimeEvolution {
    steps: Vec<TimeEvolutionStep>,
}

#[derive(Debug)]
enum TimeEvolutionStep {
    Coupling(Vec<i64>, Tensor),
    Polarization(PolarizationGate),
}

impl ADQCTimeEvolution {
    /// Construct time-evolution circuit.
    pub fn new(
        vs: &nn::Path<'_>,
        hamiltonian: &Tensor,
        num_qubits: i64,
        time_steps: i64,
        time_slice: f64,
        directions: &[SpinDirection],
    ) -> Self {
        let h_shape = hamiltonian.size();
        assert!(h_shape == [4, 4] || h_shape == [2, 2, 2, 2]);
        assert!(num_qubits > 0);
        assert!(time_steps > 0);
        assert!(time_slice > 0.0);
        let h = if h_shape == [2, 2, 2, 2] {
            hamiltonian.reshape([4, 4])
        } else {
            hamiltonian.shallow_clone()
        };
        let imag = Tensor::complex(
            &Tensor::zeros([], (Kind::Float, h.device())),
            &Tensor::ones([], (Kind::Float, h.device())),
        );
        let u = (imag * (-time_slice) * h).matrix_exp();
        let u = view_gate_matrix_as_tensor(&u, Some(2));
        let per_layer = ADQCNet::calc_gate_target_qubit_positions(GatePattern::Brick, num_qubits);
        let mut steps = Vec::new();
        for t in 0..time_steps {
            for position in &per_layer {
                steps.push(TimeEvolutionStep::Coupling(
                    position.clone(),
                    u.shallow_clone(),
                ));
            }
            for qubit_idx in 0..num_qubits {
                steps.push(TimeEvolutionStep::Polarization(PolarizationGate::new(
                    &vs.sub(format!("polarization_{t}_{qubit_idx}")),
                    time_slice,
                    qubit_idx,
                    directions,
                )));
            }
        }
        Self { steps }
    }

    /// Forward pass.
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let mut state = tensor.shallow_clone();
        for step in &self.steps {
            state = match step {
                TimeEvolutionStep::Coupling(target, gate) => apply_gate(&state, gate, target, &[]),
                TimeEvolutionStep::Polarization(gate) => gate.forward(&state),
            };
        }
        state
    }
}
