//! Thin gate wrappers around functional gate application.

use tch::{Device, Kind, Tensor, nn};

use crate::tensor_gates::functional::{
    Pauli, apply_gate, apply_gate_batched, pauli_operator, rotate_from_params,
};
use crate::utils::checking::check_quantum_gate;
use crate::utils::mapping::{view_gate_matrix_as_tensor, view_gate_tensor_as_matrix};

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

/// Trainable ADQC gate parameterized by a latent complex tensor.
#[derive(Debug)]
pub struct ADQCGate {
    gate_real: Tensor,
    gate_imag: Tensor,
    batched_input: bool,
    target_qubit: Vec<i64>,
    control_qubit: Vec<i64>,
    double_precision: bool,
}

impl ADQCGate {
    /// Construct an ADQC gate under an `nn::Path`.
    pub fn new(
        vs: &nn::Path<'_>,
        target_qubit: Vec<i64>,
        control_qubit: Vec<i64>,
        batched_input: bool,
        double_precision: bool,
        identity_init: bool,
    ) -> Self {
        assert!(!target_qubit.is_empty(), "target_qubit must be non-empty");
        let dims = vec![2; target_qubit.len() * 2];
        let device = vs.device();
        let mut gate_real = Tensor::randn(dims.as_slice(), (Kind::Float, device));
        let mut gate_imag = Tensor::randn(dims.as_slice(), (Kind::Float, device));
        if identity_init {
            let identity = Tensor::eye(2_i64.pow(target_qubit.len() as u32), (Kind::Float, device))
                .reshape(dims.as_slice());
            gate_real = identity + 0.001 * gate_real;
            gate_imag *= 0.001;
        }
        if double_precision {
            gate_real = gate_real.to_kind(Kind::Double);
            gate_imag = gate_imag.to_kind(Kind::Double);
        }
        let gate_real = vs.add("gate_real", gate_real, true);
        let gate_imag = vs.add("gate_imag", gate_imag, true);
        Self {
            gate_real,
            gate_imag,
            batched_input,
            target_qubit,
            control_qubit,
            double_precision,
        }
    }

    fn gate_params(&self) -> Tensor {
        let real = if self.double_precision {
            self.gate_real.to_kind(Kind::Double)
        } else {
            self.gate_real.shallow_clone()
        };
        let imag = if self.double_precision {
            self.gate_imag.to_kind(Kind::Double)
        } else {
            self.gate_imag.shallow_clone()
        };
        Tensor::complex(&real, &imag)
    }

    /// Return the unitary gate tensor derived from latent parameters.
    pub fn gate(&self) -> Tensor {
        let params = self.gate_params();
        let (u, _s, vh) = Tensor::linalg_svd(&view_gate_tensor_as_matrix(&params, None), false, "");
        view_gate_matrix_as_tensor(&u.matmul(&vh), None)
    }
}

impl QuantumGate for ADQCGate {
    fn forward(
        &self,
        tensor: &Tensor,
        target_qubit: Option<&[i64]>,
        control_qubit: Option<&[i64]>,
    ) -> Tensor {
        let gate = self.gate();
        let target = target_qubit.unwrap_or(&self.target_qubit);
        let control = control_qubit.unwrap_or(&self.control_qubit);
        if self.batched_input {
            apply_gate_batched(tensor, &gate, target, control)
        } else {
            apply_gate(tensor, &gate, target, control)
        }
    }
}

/// Build a non-trainable simple gate from a tensor on a target device.
pub fn simple_gate_on_device(
    gate: &Tensor,
    batched_input: bool,
    target_qubit: Vec<i64>,
    device: Device,
) -> SimpleGate {
    SimpleGate::new(
        gate.to_device(device),
        batched_input,
        Some(target_qubit),
        None,
        Some(false),
    )
}
