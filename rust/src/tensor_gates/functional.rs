//! Functional quantum-gate operations.

use tch::{Device, IndexOp, Kind, Tensor};

use crate::utils::checking::{check_quantum_gate, check_state_tensor, iterable_have_common};
use crate::utils::mapping::{inverse_permutation, map_float_kind_to_complex, unify_tensor_dtypes};

/// Apply a quantum gate to a dense quantum-state tensor.
pub fn apply_gate(
    quantum_state: &Tensor,
    gate: &Tensor,
    target_qubit: &[i64],
    control_qubit: &[i64],
) -> Tensor {
    check_state_tensor(quantum_state);
    assert!(
        !iterable_have_common(target_qubit, control_qubit),
        "target qubit and control qubit must not overlap"
    );
    let num_qubits = quantum_state.dim() as i64;
    assert!(
        num_qubits >= target_qubit.len() as i64 + control_qubit.len() as i64,
        "number of qubits must be greater than or equal to the number of target qubits and control qubits"
    );
    check_quantum_gate(gate, Some(target_qubit.len() as i64), false);
    let (state, mut gate) = unify_tensor_dtypes(quantum_state, gate);
    gate = gate.to_device(state.device());
    for &qidx in target_qubit {
        assert!(
            0 <= qidx && qidx < num_qubits,
            "target qubit index {qidx} out of range"
        );
    }
    for &qidx in control_qubit {
        assert!(
            0 <= qidx && qidx < num_qubits,
            "control qubit index {qidx} out of range"
        );
    }
    let target_count = target_qubit.len() as i64;
    if gate.dim() == 2 {
        gate = gate.reshape(vec![2; (target_count * 2) as usize]);
    }
    let mut other_qubits: Vec<i64> = (0..num_qubits).collect();
    other_qubits.retain(|idx| !target_qubit.contains(idx) && !control_qubit.contains(idx));
    let permutation: Vec<i64> = target_qubit
        .iter()
        .copied()
        .chain(other_qubits.iter().copied())
        .chain(control_qubit.iter().copied())
        .collect();
    let state = state.permute(&permutation);
    let state_shape = state.size();
    let mut new_shape = vec![2; target_qubit.len() + other_qubits.len()];
    new_shape.push(-1);
    let state = state.reshape(new_shape);
    let control_flat = state.size()[state.dim() - 1];
    let unaffected = if control_flat > 1 {
        Some(state.narrow(-1, 0, control_flat - 1))
    } else {
        None
    };
    let state_to_apply = state.select(-1, control_flat - 1);
    let labels: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .chars()
        .collect();
    let needed = target_qubit.len() * 2 + other_qubits.len();
    assert!(
        needed <= labels.len(),
        "too many qubits for compact einsum labels"
    );
    let target_labels = &labels[0..target_qubit.len()];
    let other_labels = &labels[target_qubit.len()..target_qubit.len() + other_qubits.len()];
    let out_labels = &labels[target_qubit.len() + other_qubits.len()..needed];
    let target_string = target_labels.iter().collect::<String>();
    let other_string = other_labels.iter().collect::<String>();
    let out_string = out_labels.iter().collect::<String>();
    let mut equation = String::new();
    equation.push_str(&out_string);
    equation.push_str(&target_string);
    equation.push(',');
    equation.push_str(&target_string);
    equation.push_str(&other_string);
    equation.push_str("->");
    equation.push_str(&out_string);
    equation.push_str(&other_string);
    let new_state = Tensor::einsum(&equation, &[&gate, &state_to_apply], None::<i64>).unsqueeze(-1);
    let final_state = match unaffected {
        Some(unaffected) => Tensor::cat(&[unaffected, new_state], -1),
        None => new_state,
    };
    let final_state = final_state.reshape(&state_shape);
    final_state.permute(inverse_permutation(&permutation))
}

/// Apply a gate to batched quantum states with batch dimension at 0.
pub fn apply_gate_batched(
    quantum_states: &Tensor,
    gate: &Tensor,
    target_qubit: &[i64],
    control_qubit: &[i64],
) -> Tensor {
    let batch = quantum_states.size()[0];
    let states = quantum_states.split(1, 0);
    let outputs: Vec<Tensor> = states
        .iter()
        .map(|state| apply_gate(&state.squeeze_dim(0), gate, target_qubit, control_qubit))
        .collect();
    assert_eq!(outputs.len() as i64, batch);
    Tensor::stack(&outputs, 0)
}

/// Kronecker product of two or more matrices.
pub fn kron(matrices: &[Tensor]) -> Tensor {
    assert!(
        matrices.len() >= 2,
        "At least two matrices are needed for Kronecker product"
    );
    matrices
        .iter()
        .skip(1)
        .fold(matrices[0].shallow_clone(), |acc, matrix| acc.kron(matrix))
}

/// Outer product of disjoint quantum gates.
pub fn gate_outer_product(gates: &[Tensor], matrix_form: bool) -> Tensor {
    assert!(gates.len() >= 2, "at least 2 gates");
    let num_qubits = gates
        .iter()
        .map(|gate| check_quantum_gate(gate, None, false))
        .collect::<Vec<_>>();
    let total_qubits = num_qubits.iter().sum::<i64>() as usize;
    let labels = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .chars()
        .collect::<Vec<_>>();
    assert!(
        total_qubits * 2 <= labels.len(),
        "too many qubits for compact einsum labels"
    );
    let mut next_label = 0;
    let mut left_labels = Vec::new();
    let mut right_labels = Vec::new();
    let mut gate_tensors = Vec::new();
    let mut input_terms = Vec::new();
    for (gate, &qubits) in gates.iter().zip(num_qubits.iter()) {
        let left = labels[next_label..next_label + qubits as usize].to_vec();
        next_label += qubits as usize;
        let right = labels[next_label..next_label + qubits as usize].to_vec();
        next_label += qubits as usize;
        let shape = vec![2; (qubits * 2) as usize];
        let gate_tensor = if gate.dim() == 2 {
            gate.reshape(shape.as_slice())
        } else {
            gate.shallow_clone()
        };
        input_terms.push(left.iter().chain(right.iter()).copied().collect::<String>());
        left_labels.extend(left);
        right_labels.extend(right);
        gate_tensors.push(gate_tensor);
    }
    let output = left_labels
        .iter()
        .chain(right_labels.iter())
        .copied()
        .collect::<String>();
    let equation = format!("{}->{}", input_terms.join(","), output);
    let gate_refs = gate_tensors.iter().collect::<Vec<_>>();
    let product = Tensor::einsum(&equation, &gate_refs, None::<i64>);
    if matrix_form {
        let dim = 2_i64.pow(total_qubits as u32);
        product.reshape([dim, dim])
    } else {
        product
    }
}

/// Pauli operator selector.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Pauli {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
    /// Identity.
    Id,
}

/// Return a Pauli operator.
pub fn pauli_operator(
    pauli: Pauli,
    double_precision: bool,
    force_complex: bool,
    device: Device,
) -> Tensor {
    let real_kind = if double_precision {
        Kind::Double
    } else {
        Kind::Float
    };
    let complex_kind = if double_precision {
        Kind::ComplexDouble
    } else {
        Kind::ComplexFloat
    };
    let kind = if force_complex {
        complex_kind
    } else {
        real_kind
    };
    match pauli {
        Pauli::X => Tensor::from_slice(&[0.0_f32, 1.0, 1.0, 0.0])
            .to_kind(kind)
            .to_device(device)
            .reshape([2, 2]),
        Pauli::Y => {
            let real = Tensor::from_slice(&[0.0_f32, 0.0, 0.0, 0.0]).reshape([2, 2]);
            let imag = Tensor::from_slice(&[0.0_f32, -1.0, 1.0, 0.0]).reshape([2, 2]);
            Tensor::complex(&real, &imag)
                .to_kind(complex_kind)
                .to_device(device)
        }
        Pauli::Z => Tensor::from_slice(&[1.0_f32, 0.0, 0.0, -1.0])
            .to_kind(kind)
            .to_device(device)
            .reshape([2, 2]),
        Pauli::Id => Tensor::eye(2, (kind, device)),
    }
}

/// Return a spin operator for a direction.
pub fn spin_operator(direction: Pauli, device: Device) -> Tensor {
    match direction {
        Pauli::Id => pauli_operator(Pauli::Id, false, false, device),
        other => pauli_operator(other, false, false, device) / 2.0,
    }
}

/// Rotation gate parameterization used by the notebooks.
pub fn rotate_from_params(params_vec: &Tensor) -> Tensor {
    assert_eq!(
        params_vec.size(),
        vec![4],
        "params must be a 4-element vector"
    );
    let kind = params_vec.kind();
    assert!(
        matches!(kind, Kind::Float | Kind::Double),
        "params must be float32 or float64"
    );
    let device = params_vec.device();
    let gate_kind = map_float_kind_to_complex(kind);
    let beta = params_vec.i(0);
    let delta = params_vec.i(1);
    let ita = params_vec.i(2);
    let gamma = params_vec.i(3);
    let beta_coeff = Tensor::from_slice(&[-0.5_f32, -0.5, 0.5, 0.5])
        .to_kind(gate_kind)
        .to_device(device)
        .reshape([2, 2]);
    let beta_matrix = &beta_coeff * beta;
    let delta_matrix = beta_coeff.transpose(0, 1) * delta;
    let gamma_half = gamma / 2.0;
    let gamma_matrix = Tensor::eye(2, (gate_kind, device)) * gamma_half.cos()
        + pauli_operator(Pauli::X, kind == Kind::Double, true, device) * gamma_half.sin();
    let coeff = Tensor::from_slice(&[1.0_f32, -1.0, 1.0, 1.0])
        .to_kind(gate_kind)
        .to_device(device)
        .reshape([2, 2]);
    let phase = (ita + beta_matrix + delta_matrix)
        * Tensor::complex(
            &Tensor::zeros([], (kind, device)),
            &Tensor::ones([], (kind, device)),
        );
    coeff * phase.exp() * gamma_matrix
}

/// Identity gate tensor.
pub fn identity_gate_tensor(
    num_qubits: i64,
    matrix_form: bool,
    kind: Kind,
    device: Device,
) -> Tensor {
    assert!(num_qubits > 0, "num_qubits must be positive");
    let matrix = Tensor::eye(2_i64.pow(num_qubits as u32), (kind, device));
    if matrix_form {
        matrix
    } else {
        let shape = vec![2; (num_qubits * 2) as usize];
        matrix.view(shape.as_slice())
    }
}

/// Controlled gate tensor.
pub fn get_control_gate_tensor(
    num_control_qubits: i64,
    applied_gate: &Tensor,
    matrix_form: bool,
    kind: Kind,
    device: Device,
) -> Tensor {
    assert!(
        num_control_qubits > 0,
        "num_control_qubits must be positive"
    );
    let num_applied = check_quantum_gate(applied_gate, None, false);
    let applied = if applied_gate.dim() > 2 {
        applied_gate.view([2_i64.pow(num_applied as u32), 2_i64.pow(num_applied as u32)])
    } else {
        applied_gate.shallow_clone()
    }
    .to_kind(kind)
    .to_device(device);
    let total = num_control_qubits + num_applied;
    let matrix = identity_gate_tensor(total, true, kind, device);
    let slice_start = (2_i64.pow(num_control_qubits as u32) - 1) << num_applied;
    let mut target = matrix.i((slice_start.., slice_start..));
    target.copy_(&applied);
    if matrix_form {
        matrix
    } else {
        let shape = vec![2; (total * 2) as usize];
        matrix.view(shape.as_slice())
    }
}

/// Generate a random unitary matrix.
pub fn rand_unitary(dim: i64, kind: Kind, device: Device) -> Tensor {
    assert!(dim > 0, "dim must be positive");
    let work_device = crate::utils::devices::linalg_work_device(device);
    let mat = Tensor::randn([dim, dim], (kind, work_device));
    let (q, _) = Tensor::linalg_qr(&mat, "reduced");
    q.to_device(device)
}

/// Generate a random gate tensor.
pub fn rand_gate_tensor(num_qubits: i64, matrix_form: bool, kind: Kind, device: Device) -> Tensor {
    let matrix = rand_unitary(2_i64.pow(num_qubits as u32), kind, device);
    if matrix_form {
        matrix
    } else {
        let shape = vec![2; (num_qubits * 2) as usize];
        matrix.view(shape.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn gate_outer_product_matches_kron_for_matrices() {
        let x = pauli_operator(Pauli::X, false, false, Device::Cpu);
        let z = pauli_operator(Pauli::Z, false, false, Device::Cpu);
        let product = gate_outer_product(&[x.shallow_clone(), z.shallow_clone()], true);
        let expected = kron(&[x, z]);
        assert!(product.allclose(&expected, 1e-6, 1e-8, false));
    }

    #[test]
    fn gate_outer_product_returns_tensor_form() {
        let x = pauli_operator(Pauli::X, false, false, Device::Cpu);
        let id = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let product = gate_outer_product(&[x, id], false);
        assert_eq!(product.size(), vec![2, 2, 2, 2, 2, 2]);
    }
}
