//! Thin gate wrappers around functional gate application.

use tch::Tensor;

use crate::tensor_gates::functional::{
    Pauli, apply_gate, apply_gate_batched, pauli_operator, rotate_from_params,
};
use crate::utils::checking::check_quantum_gate;

/// Common gate interface.
pub trait QuantumGate: std::fmt::Debug + Send {
    /// Apply this gate to an input tensor.
    fn forward(
        &self,
        tensor: &Tensor,
        target_qubit: Option<&[i64]>,
        control_qubit: Option<&[i64]>,
    ) -> Tensor;
}

/// Simple fixed gate.
#[derive(Debug)]
pub struct SimpleGate {
    gate: Tensor,
    batched_input: bool,
    target_qubit: Option<Vec<i64>>,
    control_qubit: Option<Vec<i64>>,
}

impl SimpleGate {
    /// Construct a simple gate.
    pub fn new(
        gate: Tensor,
        batched_input: bool,
        target_qubit: Option<Vec<i64>>,
        control_qubit: Option<Vec<i64>>,
        requires_grad: Option<bool>,
    ) -> Self {
        check_quantum_gate(&gate, None, false);
        if let Some(requires_grad) = requires_grad {
            let _ = gate.set_requires_grad(requires_grad);
        }
        Self {
            gate,
            batched_input,
            target_qubit,
            control_qubit,
        }
    }
}

impl QuantumGate for SimpleGate {
    fn forward(
        &self,
        tensor: &Tensor,
        target_qubit: Option<&[i64]>,
        control_qubit: Option<&[i64]>,
    ) -> Tensor {
        let target = target_qubit
            .or(self.target_qubit.as_deref())
            .expect("target_qubit must be specified or set in the gate");
        let control = control_qubit
            .or(self.control_qubit.as_deref())
            .unwrap_or(&[]);
        if self.batched_input {
            apply_gate_batched(tensor, &self.gate, target, control)
        } else {
            apply_gate(tensor, &self.gate, target, control)
        }
    }
}

/// Fixed Pauli gate.
#[derive(Debug)]
pub struct PauliGate {
    is_identity: bool,
    inner: SimpleGate,
}

impl PauliGate {
    /// Construct a Pauli gate.
    pub fn new(
        pauli: Pauli,
        batched_input: bool,
        target_qubit: Option<Vec<i64>>,
        control_qubit: Option<Vec<i64>>,
    ) -> Self {
        let gate = pauli_operator(pauli, false, false, tch::Device::Cpu);
        Self {
            is_identity: pauli == Pauli::Id,
            inner: SimpleGate::new(
                gate,
                batched_input,
                target_qubit,
                control_qubit,
                Some(false),
            ),
        }
    }
}

impl QuantumGate for PauliGate {
    fn forward(
        &self,
        tensor: &Tensor,
        target_qubit: Option<&[i64]>,
        control_qubit: Option<&[i64]>,
    ) -> Tensor {
        if self.is_identity {
            tensor.shallow_clone()
        } else {
            self.inner.forward(tensor, target_qubit, control_qubit)
        }
    }
}

/// Rotation gate backed by a trainable parameter tensor.
#[derive(Debug)]
pub struct RotateGate {
    params: Tensor,
    batched_input: bool,
    target_qubit: Option<Vec<i64>>,
    control_qubit: Option<Vec<i64>>,
}

impl RotateGate {
    /// Construct from a parameter tensor of shape `[4]`.
    pub fn new(
        params: Tensor,
        batched_input: bool,
        target_qubit: Option<Vec<i64>>,
        control_qubit: Option<Vec<i64>>,
    ) -> Self {
        assert_eq!(params.size(), vec![4], "params must be a 4-element vector");
        let _ = params.set_requires_grad(true);
        Self {
            params,
            batched_input,
            target_qubit,
            control_qubit,
        }
    }
}

impl QuantumGate for RotateGate {
    fn forward(
        &self,
        tensor: &Tensor,
        target_qubit: Option<&[i64]>,
        control_qubit: Option<&[i64]>,
    ) -> Tensor {
        let gate = rotate_from_params(&self.params);
        let target = target_qubit
            .or(self.target_qubit.as_deref())
            .expect("target_qubit must be specified or set in the gate");
        let control = control_qubit
            .or(self.control_qubit.as_deref())
            .unwrap_or(&[]);
        if self.batched_input {
            apply_gate_batched(tensor, &gate, target, control)
        } else {
            apply_gate(tensor, &gate, target, control)
        }
    }
}
