use crate::utils::einsum::named_einsum;
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

/// Outer product of 1D vectors producing an N-order tensor with shape [d1, d2, ..., dN].
pub fn gate_outer_product(vectors: &[Tensor]) -> Tensor {
    assert!(
        vectors.len() >= 2,
        "At least two vectors are required for outer product"
    );
    for (i, v) in vectors.iter().enumerate() {
        assert!(v.dim() == 1, "Expected 1D tensor at index {}", i);
    }
    let mut out = vectors[0].shallow_clone();
    for (i, v) in vectors.iter().enumerate().skip(1) {
        let a = out.unsqueeze(-1);
        let mut b = v.shallow_clone();
        for _ in 0..i {
            b = b.unsqueeze(0);
        }
        out = a * b; // broadcast multiplies to grow dimensionality
    }
    out
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
            // Y = [[0, -i], [i, 0]]
            let k_complex = if double_precision {
                Kind::ComplexDouble
            } else {
                Kind::ComplexFloat
            };
            // re = [[0,0],[0,0]], im = [[0,-1],[1,0]]
            let re = [0.0, 0.0, 0.0, 0.0];
            let im = [0.0, -1.0, 1.0, 0.0];
            crate::utils::complex_from_slices(&re, &im, &[2, 2], k_complex)
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

/// Heisenberg/Pauli spin operator along a direction.
/// Note: Y requires complex support. TODO: implement complex constants and return proper Y.
pub fn spin_operator(direction: &str) -> Tensor {
    match direction {
        "X" | "Z" | "ID" => pauli_operator(direction, false, false) / 2.0,
        "Y" => pauli_operator("Y", false, true) / 2.0,
        _ => panic!("direction must be one of X, Y, Z, ID"),
    }
}

pub fn heisenberg(
    jx: f64,
    jy: f64,
    jz: f64,
    double_precision: bool,
    return_matrix: bool,
) -> Tensor {
    let px = pauli_operator("X", double_precision, true);
    let py = pauli_operator("Y", double_precision, true);
    let pz = pauli_operator("Z", double_precision, true);
    // h = jx X⊗X + jy Y⊗Y + jz Z⊗Z
    let xx = Tensor::einsum(
        "ab, ij -> aibj",
        &[px.shallow_clone(), px],
        None::<Vec<i64>>,
    );
    let yy = Tensor::einsum(
        "ab, ij -> aibj",
        &[py.shallow_clone(), py],
        None::<Vec<i64>>,
    );
    let zz = Tensor::einsum(
        "ab, ij -> aibj",
        &[pz.shallow_clone(), pz],
        None::<Vec<i64>>,
    );
    let mut h = &xx * jx + &yy * jy + &zz * jz;
    h = &h / 4.0;
    if return_matrix { h.view([4, 4]) } else { h }
}

/// Rotation gate from scalars (ita, beta, delta, gamma) as in the notebook.
pub fn rotate_from_scalars(
    ita: f64,
    beta: f64,
    delta: f64,
    gamma: f64,
    double_precision: bool,
) -> Tensor {
    let k_complex = if double_precision {
        Kind::ComplexDouble
    } else {
        Kind::ComplexFloat
    };
    let k_real = if double_precision {
        Kind::Double
    } else {
        Kind::Float
    };

    // Coefficient matrices (complex dtype but purely real values)
    let beta_coeff = Tensor::f_from_slice(&[-0.5, -0.5, 0.5, 0.5])
        .unwrap()
        .to_kind(k_real)
        .view([2, 2])
        .to_kind(k_complex);
    let delta_coeff = beta_coeff.tr();

    let beta_t = Tensor::from(beta).to_kind(k_real);
    let delta_t = Tensor::from(delta).to_kind(k_real);
    let ita_t = Tensor::from(ita).to_kind(k_real);
    let gamma_t = Tensor::from(gamma).to_kind(k_real);

    let beta_mat = &beta_coeff * beta_t;
    let delta_mat = &delta_coeff * delta_t;

    // gamma block: cos(gamma/2) * I + sin(gamma/2) * X
    let gamma_2 = &gamma_t / 2.0;
    let eye = Tensor::eye(2, (k_complex, Device::Cpu));
    let x = pauli_operator("X", double_precision, true);
    let gamma_mat = eye * gamma_2.cos().to_kind(k_complex) + x * gamma_2.sin().to_kind(k_complex);

    let coefficient = Tensor::f_from_slice(&[1.0, -1.0, 1.0, 1.0])
        .unwrap()
        .to_kind(k_real)
        .view([2, 2])
        .to_kind(k_complex);

    // exp(i * (ita + beta + delta)) = cos(phase) + i sin(phase)
    let phase = ita_t + beta_mat.real() + delta_mat.real();
    let phase_cos = phase.cos().to_kind(k_real).to_kind(k_complex);
    let phase_sin = phase.sin().to_kind(k_real);
    let exp_i_phase = Tensor::f_complex(&phase_cos.to_kind(k_real), &phase_sin).unwrap();
    coefficient * exp_i_phase * gamma_mat
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

    // Gate to tensor form [g1..gt, t1..tt]
    let d_t = 1_i64 << num_target;
    let gate_t = if gate_u.dim() == 2 {
        let dims = vec![2_i64; (2 * num_target) as usize];
        gate_u.view(&dims[..])
    } else {
        gate_u.shallow_clone()
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
    let last_dims = vec![2_i64; (num_target + (other.len() as i64)) as usize];
    let last_as_tensor = last_col.view(&last_dims[..]);
    // Use named einsum: (g t, t o) -> (g o)
    let g_names: Vec<String> = (0..num_target).map(|i| format!("g{}", i)).collect();
    let t_names: Vec<String> = (0..num_target).map(|i| format!("t{}", i)).collect();
    let o_names: Vec<String> = (0..(other.len() as i64))
        .map(|i| format!("o{}", i))
        .collect();
    let gate_dims = format!("{} {}", g_names.join(" "), t_names.join(" "));
    let state_dims = format!("{} {}", t_names.join(" "), o_names.join(" "));
    let out_dims = format!("{} {}", g_names.join(" "), o_names.join(" "));
    let spec = format!(
        "{}, {} -> {}",
        gate_dims.trim(),
        state_dims.trim(),
        out_dims.trim()
    );
    let new_last_t = named_einsum(&spec, &[gate_t.shallow_clone(), last_as_tensor]);
    let new_last = new_last_t.view([d_t * d_o, 1]);
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

/// Batched version by simple looping over batch dim (TODO: vectorize/einsum/vmap).
pub fn apply_gate_batched(
    quantum_states: &Tensor, // [B, 2, 2, ..., 2]
    gate: &Tensor,
    target_qubit: Vec<i64>,
    control_qubit: Option<Vec<i64>>,
) -> Tensor {
    let b = quantum_states.size()[0];
    let mut outs: Vec<Tensor> = Vec::with_capacity(b as usize);
    for i in 0..b {
        let s = quantum_states.i(i);
        let out = apply_gate(&s, gate, target_qubit.clone(), control_qubit.clone());
        outs.push(out);
    }
    Tensor::stack(&outs, 0)
}

/// Placeholder for vmap-style API. TODO: replace looping with a vectorized contraction.
pub fn apply_gate_batched_with_vmap(
    quantum_states: &Tensor,
    gate: &Tensor,
    target_qubit: Vec<i64>,
    control_qubit: Option<Vec<i64>>,
) -> Tensor {
    // TODO: use a single einsum/batched matmul once API is stabilized or added.
    apply_gate_batched(quantum_states, gate, target_qubit, control_qubit)
}

/// Random unitary via Gram-Schmidt orthogonalization (fallback without linalg::qr bindings).
pub fn rand_unitary(dim: i64, kind: Option<Kind>) -> Tensor {
    let k = kind.unwrap_or(default_float_kind());
    let m = Tensor::randn([dim, dim], (k, Device::Cpu));
    let mut q_cols: Vec<Tensor> = Vec::with_capacity(dim as usize);
    for j in 0..dim {
        let mut v = m.i((.., j));
        for qc in &q_cols {
            let proj = (&v * qc).sum(k) / (qc.square().sum(k) + 1e-12);
            v = &v - &(qc * proj);
        }
        let n = v.norm();
        let v = &v / (n + 1e-12);
        q_cols.push(v);
    }
    Tensor::stack(&q_cols, 1)
}

pub fn rand_gate_tensor(num_qubits: i64, matrix_form: bool, kind: Option<Kind>) -> Tensor {
    let dim = 1_i64 << num_qubits;
    let u = rand_unitary(dim, kind);
    if matrix_form {
        u
    } else {
        let dims = vec![2_i64; (num_qubits * 2) as usize];
        u.view(&dims[..])
    }
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
    fn test_gate_outer_product_shapes() {
        let v1 = Tensor::f_from_slice(&[1.0, 2.0]).unwrap();
        let v2 = Tensor::f_from_slice(&[3.0, 4.0, 5.0]).unwrap();
        let v3 = Tensor::f_from_slice(&[6.0]).unwrap();
        let t = gate_outer_product(&[v1, v2, v3]);
        assert_eq!(t.size(), vec![2, 3, 1]);
        // check a value: t[1,2,0] == 2 * 5 * 6
        let val = t.double_value(&[1, 2, 0]);
        assert!((val - 60.0).abs() < 1e-12);
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

    #[test]
    fn test_pauli_y_square_identity() {
        let y = pauli_operator("Y", true, true);
        let yy = y.matmul(&y);
        let i = Tensor::eye(2, (y.kind(), y.device()));
        let diff = (yy - i).abs().sum(y.kind()).double_value(&[]);
        assert!(diff < 1e-10);
    }

    #[test]
    fn test_rotate_from_scalars_basic() {
        let g = rotate_from_scalars(0.1, 0.2, -0.3, 0.4, true);
        assert_eq!(g.size(), vec![2, 2]);
        let s = g.abs().sum(g.kind()).double_value(&[]);
        assert!(s.is_finite());
    }
}
