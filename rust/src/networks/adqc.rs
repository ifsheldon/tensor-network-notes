//! Automatically differentiable quantum-circuit networks.

use tch::{Kind, Tensor, nn};

use crate::tensor_gates::modules::{ADQCGate, QuantumGate};
use crate::types::GatePattern;

/// A simple ADQC network.
#[derive(Debug)]
pub struct ADQCNet {
    gates: Vec<ADQCGate>,
    num_qubits: i64,
}

impl ADQCNet {
    /// Construct an ADQC network.
    pub fn new(
        vs: &nn::Path<'_>,
        num_qubits: i64,
        num_layers: i64,
        gate_pattern: GatePattern,
        identity_init: bool,
        double_precision: bool,
    ) -> Self {
        assert!(num_qubits > 0, "number of qubits must be greater than 0");
        assert!(num_layers > 0, "num_layers must be greater than 0");
        let positions = Self::calc_gate_target_qubit_positions(gate_pattern, num_qubits);
        let mut gates = Vec::new();
        for layer_idx in 0..num_layers {
            for (gate_idx, target) in positions.iter().enumerate() {
                let path = vs.sub(format!("adqc_{layer_idx}_{gate_idx}"));
                gates.push(ADQCGate::new(
                    &path,
                    target.clone(),
                    Vec::new(),
                    true,
                    double_precision,
                    identity_init,
                ));
            }
        }
        Self { gates, num_qubits }
    }

    /// Forward pass.
    pub fn forward(&self, qubit_states: &Tensor) -> Tensor {
        assert_eq!(
            qubit_states.dim() as i64,
            self.num_qubits + 1,
            "qubit_states must have num_qubits + 1 dimensions"
        );
        self.gates
            .iter()
            .fold(qubit_states.shallow_clone(), |state, gate| {
                gate.forward(&state, None, None)
            })
    }

    /// Calculate target qubit positions for a layer.
    pub fn calc_gate_target_qubit_positions(
        gate_pattern: GatePattern,
        num_qubits: i64,
    ) -> Vec<Vec<i64>> {
        assert!(num_qubits > 0, "number of qubits must be greater than 0");
        let mut positions = Vec::new();
        match gate_pattern {
            GatePattern::Stair => {
                for p in 0..num_qubits - 1 {
                    positions.push(vec![p, p + 1]);
                }
            }
            GatePattern::Brick => {
                let mut p = 0;
                while p < num_qubits - 1 {
                    positions.push(vec![p, p + 1]);
                    p += 2;
                }
                p = 1;
                while p < num_qubits - 1 {
                    positions.push(vec![p, p + 1]);
                    p += 2;
                }
            }
        }
        positions
    }
}

/// Compute class probabilities from ADQC output states.
pub fn probabilities_adqc_classifier(
    qubit_states: &Tensor,
    num_classes: i64,
    fast_mode: bool,
) -> Tensor {
    const DELTA: f64 = 1e-10;
    assert!(num_classes >= 2, "number of classes must be greater than 2");
    let required = ((num_classes as f64).log2().ceil()) as i64;
    assert!(qubit_states.dim() as i64 > required);
    let batch = qubit_states.size()[0];
    let states = qubit_states.reshape([batch, -1, 2_i64.pow(required as u32)]);
    let substates = states.narrow(2, 0, num_classes);
    let probs = if fast_mode {
        substates.norm_scalaropt_dim(2.0, [1].as_slice(), false)
    } else {
        (substates.shallow_clone() * substates.conj())
            .real()
            .sum_dim_intlist([1].as_slice(), false, None::<Kind>)
    };
    let norm = probs.sum_dim_intlist([1].as_slice(), true, None::<Kind>) + DELTA;
    probs / norm
}

/// Calculate classification accuracy.
pub fn calc_accuracy(probabilities: &Tensor, targets: &Tensor) -> f64 {
    assert_eq!(probabilities.dim(), 2);
    assert_eq!(targets.dim(), 1);
    assert_eq!(probabilities.size()[0], targets.size()[0]);
    probabilities
        .argmax(1, false)
        .eq_tensor(targets)
        .to_kind(Kind::Float)
        .mean(None::<Kind>)
        .double_value(&[])
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn brick_positions_match_python_convention() {
        assert_eq!(
            ADQCNet::calc_gate_target_qubit_positions(GatePattern::Brick, 5),
            vec![vec![0, 1], vec![2, 3], vec![1, 2], vec![3, 4]]
        );
    }

    #[test]
    fn probabilities_are_normalized() {
        let state = Tensor::ones([2, 2, 2], (Kind::ComplexFloat, Device::Cpu));
        let probs = probabilities_adqc_classifier(&state, 2, false);
        assert_eq!(probs.size(), vec![2, 2]);
        let sums = probs.sum_dim_intlist([1].as_slice(), false, None::<Kind>);
        assert!(sums.allclose(
            &Tensor::ones([2], (Kind::Float, Device::Cpu)),
            1e-5,
            1e-8,
            false
        ));
    }
}
