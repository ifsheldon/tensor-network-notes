use crate::utils::mapping::{
    map_float_kind_to_complex, unify_tensor_dtypes, view_gate_matrix_as_tensor,
};
use crate::utils::{checking::check_quantum_gate, default_float_kind};
use tch::{Device, IndexOp, Kind, Tensor};

/// Kronecker product of two matrices.
pub fn kron2(a: &Tensor, b: &Tensor) -> Tensor {
    // (m x n) ⊗ (p x q) = (m*p) x (n*q)
    let a_sz = a.size();
    let b_sz = b.size();
    assert!(a_sz.len() == 2 && b_sz.len() == 2, "kron expects matrices");
    let (m, n) = (a_sz[0], a_sz[1]);
    let (p, q) = (b_sz[0], b_sz[1]);
    let a_exp = a.unsqueeze(1).unsqueeze(3); // m x 1 x n x 1
    let b_exp = b.unsqueeze(0).unsqueeze(2); // 1 x p x 1 x q
    let prod = a_exp * b_exp; // m x p x n x q (broadcasted)
    prod.view([m * p, n * q])
}

/// Kronecker product of many matrices.
pub fn kron(mats: &[Tensor]) -> Tensor {
    assert!(
        mats.len() >= 2,
        "At least two matrices are required for kron"
    );
    let mut acc = mats[0].shallow_clone();
    for m in &mats[1..] {
        acc = kron2(&acc, m);
    }
    acc
}

/// Pauli operators X, Y, Z and identity. Y is TODO until convenient complex constructors are available.
pub fn pauli_operator(pauli: &str, double_precision: bool, force_complex: bool) -> Tensor {
    let mut kind = if double_precision {
        Kind::Double
    } else {
        Kind::Float
    };
    if force_complex {
        kind = map_float_kind_to_complex(kind);
    }
    match pauli {
        "X" => Tensor::f_from_slice(&[0.0, 1.0, 1.0, 0.0])
            .unwrap()
            .to_kind(kind)
            .view([2, 2]),
        "Z" => Tensor::f_from_slice(&[1.0, 0.0, 0.0, -1.0])
            .unwrap()
            .to_kind(kind)
            .view([2, 2]),
        "ID" => Tensor::eye(2, (kind, Device::Cpu)),
        "Y" => {
            // TODO: implement complex Y = [[0,-i],[i,0]] once convenient complex constructors are available in tch-rs.
            // Temporary placeholder: raise panic to make absence explicit.
            panic!("Pauli-Y requires complex support; TODO in port_plan.md")
        }
        _ => panic!("pauli must be one of X, Y, Z, ID"),
    }
}

pub fn identity_gate_tensor(num_qubits: i64, matrix_form: bool, kind: Option<Kind>) -> Tensor {
    let k = kind.unwrap_or(default_float_kind());
    let d = 1_i64 << num_qubits;
    if matrix_form {
        Tensor::eye(d, (k, Device::Cpu))
    } else {
        {
            let dims = vec![2_i64; (2 * num_qubits) as usize];
            Tensor::eye(d, (k, Device::Cpu)).view(&dims[..])
        }
    }
}

pub fn get_control_gate_tensor(
    num_control_qubits: i64,
    applied_gate: &Tensor,
    matrix_form: bool,
) -> Tensor {
    let nq = check_quantum_gate(applied_gate, None, false).expect("invalid gate");
    let t = if applied_gate.dim() > 2 {
        let d = 1_i64 << nq;
        applied_gate.view([d, d])
    } else {
        applied_gate.shallow_clone()
    };
    let k = applied_gate.kind();
    let d_c = 1_i64 << num_control_qubits;
    let d_t = 1_i64 << nq;
    let eye_c = Tensor::eye(d_c, (k, Device::Cpu));
    let eye_t = Tensor::eye(d_t, (k, Device::Cpu));
    // Projection onto |11..1>
    let mut e = Tensor::zeros([d_c], (k, Device::Cpu));
    let one = Tensor::from(1.0).to_kind(k);
    e = e
        .f_index_put_(&[Some(Tensor::from(d_c - 1))], &one, false)
        .unwrap();
    let p = e.unsqueeze(1).matmul(&e.unsqueeze(0));
    let u = kron(&[eye_c.shallow_clone(), eye_t.shallow_clone()])
        + kron(&[p.shallow_clone(), &t - &eye_t]);
    if matrix_form {
        u
    } else {
        view_gate_matrix_as_tensor(&u, Some(num_control_qubits + nq))
    }
}

/// Apply a (possibly controlled) gate to a non-batched quantum state tensor.
pub fn apply_gate(
    quantum_state: &Tensor,
    gate: &Tensor,
    mut target_qubit: Vec<i64>,
    control_qubit: Option<Vec<i64>>,
) -> Tensor {
    super::super::utils::checking::check_state_tensor(quantum_state).expect("invalid state");
    let mut control_qubit = control_qubit.unwrap_or_default();
    assert!(target_qubit.iter().all(|&q| q >= 0));
    assert!(control_qubit.iter().all(|&q| q >= 0));

    // Validate indices
    let num_qubits = quantum_state.dim() as i64;
    let num_target = target_qubit.len() as i64;
    let num_control = control_qubit.len() as i64;
    assert!(num_qubits >= num_target + num_control);
    check_quantum_gate(gate, Some(num_target), false).expect("invalid gate");

    // Ensure dtypes compatible
    let (state_u, gate_u) = unify_tensor_dtypes(quantum_state, gate);

    // Move dims into [targets, others, controls]
    target_qubit.sort_unstable();
    control_qubit.sort_unstable();
    let mut other: Vec<i64> = (0..num_qubits).collect();
    for &q in &target_qubit {
        other.retain(|&x| x != q);
    }
    for &q in &control_qubit {
        other.retain(|&x| x != q);
    }
    let mut perm = target_qubit.clone();
    perm.extend(other.iter());
    perm.extend(control_qubit.iter());
    let state = state_u.permute(&perm);
    let state_shape = state.size();

    // Gate to matrix (2^t x 2^t)
    let d_t = 1_i64 << num_target;
    let gate_m = if gate_u.dim() == 2 {
        gate_u.shallow_clone()
    } else {
        gate_u.view([d_t, d_t])
    };

    // Reshape to [2^t * 2^o, 2^c]
    let d_o = 1_i64 << (other.len() as i64);
    let d_c = 1_i64 << num_control;
    let state2 = state.view([d_t * d_o, d_c]);
    let cols = state2.size()[1];
    assert_eq!(cols, d_c);
    let unaffected = if d_c > 1 {
        state2.i((.., 0..(d_c - 1)))
    } else {
        Tensor::zeros([d_t * d_o, 0], (state2.kind(), state2.device()))
    };
    let last_col = state2.i((.., d_c - 1)); // shape: [d_t*d_o]
    let last_as_matrix = last_col.view([d_t, d_o]);
    let new_last = gate_m.matmul(&last_as_matrix).view([d_t * d_o, 1]);
    let final2 = if d_c > 1 {
        Tensor::cat(&[unaffected, new_last], 1)
    } else {
        new_last
    };
    // Back to original shape
    let mut new_shape = vec![2_i64; (num_target + (other.len() as i64)) as usize];
    new_shape.push(d_c);
    let final_state = final2.view(&new_shape[..]).view(&state_shape[..]);
    // Inverse permutation
    let mut inv = vec![0_i64; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as i64;
    }
    final_state.permute(&inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kron_basic() {
        let a = Tensor::f_from_slice(&[1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .view([2, 2]);
        let b = Tensor::f_from_slice(&[0.0, 5.0, 6.0, 7.0])
            .unwrap()
            .view([2, 2]);
        let k = kron(&[a, b]);
        assert_eq!(k.size(), vec![4, 4]);
    }

    #[test]
    fn test_control_gate_tensor_matrix_form() {
        let x = pauli_operator("X", false, false);
        let u = get_control_gate_tensor(1, &x, true);
        // On |11> acts as X, otherwise identity
        let v = Tensor::f_from_slice(&[1.0, 0.0, 0.0, 1.0])
            .unwrap()
            .view([2, 2]);
        // check diagonal ones at top-left 2x2
        assert!(u.i((0..2, 0..2)).eq_tensor(&v).all().int64_value(&[]) != 0);
    }
}
