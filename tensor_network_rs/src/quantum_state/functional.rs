use crate::utils::checking::{check_quantum_gate, check_state_tensor};
use crate::utils::einsum::named_einsum;
use tch::{Kind, Tensor};

pub fn calc_reduced_density_matrix(state: &Tensor, qubit_idx: Vec<i64>) -> Tensor {
    // Normalize qubit list
    let keep = qubit_idx;
    if keep.is_empty() {
        panic!("qubit_idx must be non-empty");
    }
    let num_qubits = state.dim() as i64;
    for &qi in &keep {
        assert!((0..num_qubits).contains(&qi), "qubit_idx out of range");
    }
    let mut keep_sorted = keep.clone();
    keep_sorted.sort_unstable();
    // Build permutation: [keep..., reduce...]
    let mut reduce: Vec<i64> = (0..num_qubits).collect();
    for &k in &keep_sorted {
        reduce.retain(|&x| x != k);
    }
    let mut perm = keep_sorted.clone();
    perm.extend(reduce.iter());
    let s = state.permute(&perm);
    let k = keep_sorted.len() as i64;
    let d_keep = 1_i64 << k;
    let d_red = 1_i64 << (num_qubits - k);
    let s2 = s.view([d_keep, d_red]);
    // ρ = S S^†
    let s_h = s2.conj().transpose(0, 1);
    s2.matmul(&s_h)
}

pub fn calc_observation(
    state: &Tensor,
    operator: &Tensor,
    qubit_idx: Vec<i64>,
    fast_mode: bool,
) -> Tensor {
    let len = qubit_idx.len() as i64;
    let rdm = calc_reduced_density_matrix(state, qubit_idx);
    let nq = check_quantum_gate(operator, None, false).expect("invalid operator");
    assert_eq!(nq, len, "operator qubit count mismatch");
    let d = 1_i64 << nq;
    let op_mat = if operator.dim() == 2 {
        operator.shallow_clone()
    } else {
        operator.view([d, d])
    };
    if fast_mode {
        // sum(ρ .* op^T)
        (rdm.shallow_clone() * op_mat.transpose(0, 1)).sum(rdm.kind())
    } else {
        let prod = rdm.matmul(&op_mat);
        prod.trace()
    }
}

pub fn calc_onsite_entanglement_entropy(
    state: &Tensor,
    qubit_idx: Option<Vec<i64>>,
    eps: f64,
) -> Tensor {
    check_state_tensor(state).expect("invalid state");
    let n = state.dim() as i64;
    let indices: Vec<i64> = match qubit_idx {
        None => (0..n).collect(),
        Some(v) if !v.is_empty() => v,
        _ => panic!("qubit_idx must be None or non-empty list"),
    };
    let mut ent: Vec<Tensor> = Vec::with_capacity(indices.len());
    for idx in indices {
        let rdm = calc_reduced_density_matrix(state, vec![idx]); // 2x2
        // Eigenvalues of 2x2 Hermitian [[a,c],[c*,b]]
        let a = rdm.double_value(&[0, 0]);
        let b = rdm.double_value(&[1, 1]);
        let c_re = rdm.real().double_value(&[0, 1]);
        let c_im = rdm.imag().double_value(&[0, 1]);
        let c_abs2 = c_re * c_re + c_im * c_im;
        let disc = ((a - b) * (a - b) + 4.0 * c_abs2).sqrt();
        let l1 = (a + b + disc) * 0.5;
        let l2 = (a + b - disc) * 0.5;
        let l = Tensor::f_from_slice(&[l1, l2]).unwrap();
        let s = -(l.copy() * (l.copy() + eps).log()).sum(Kind::Float);
        ent.push(s);
    }
    Tensor::stack(&ent, 0)
}

pub fn project_state(
    state: &Tensor,
    project_qubit_state: &Tensor,
    project_qubit_idx: i64,
) -> Tensor {
    check_state_tensor(state).expect("invalid state");
    assert!(project_qubit_state.dim() == 1 && project_qubit_state.size()[0] == 2);
    let n = state.dim() as i64;
    assert!((0..n).contains(&project_qubit_idx));
    // Build names s0..s{n-1}; contract dimension s{idx} with v(s)
    let s_names: Vec<String> = (0..n).map(|i| format!("s{}", i)).collect();
    let state_spec = s_names.join(" ");
    let v_spec = s_names[project_qubit_idx as usize].to_string();
    let out_spec = {
        let out: Vec<String> = s_names
            .iter()
            .enumerate()
            .filter(|(i, _)| (*i as i64) != project_qubit_idx)
            .map(|(_, s)| s.clone())
            .collect();
        out.join(" ")
    };
    let spec = format!("{}, {} -> {}", state_spec, v_spec, out_spec);
    let new_state = named_einsum(
        &spec,
        &[state.shallow_clone(), project_qubit_state.shallow_clone()],
    );
    let norm = new_state.norm();
    &new_state / norm
}

pub fn observe_bond_energies(
    quantum_state: &Tensor,
    hamiltonian: Vec<Tensor>,
    interaction_positions: Vec<Vec<i64>>,
) -> Tensor {
    check_state_tensor(quantum_state).expect("invalid state");
    assert_eq!(hamiltonian.len(), interaction_positions.len());
    let mut vals: Vec<Tensor> = Vec::with_capacity(hamiltonian.len());
    for (h, pos) in hamiltonian.iter().zip(interaction_positions.iter()) {
        check_quantum_gate(h, None, true).expect("invalid hamiltonian tensor form");
        vals.push(calc_observation(quantum_state, h, pos.clone(), true));
    }
    Tensor::stack(&vals, 0)
}

pub fn observe_bond_energies_single(
    quantum_state: &Tensor,
    hamiltonian: &Tensor,
    interaction_positions: &[Vec<i64>],
) -> Tensor {
    check_state_tensor(quantum_state).expect("invalid state");
    check_quantum_gate(hamiltonian, None, true).expect("invalid hamiltonian tensor form");
    let mut vals: Vec<Tensor> = Vec::with_capacity(interaction_positions.len());
    for pos in interaction_positions {
        vals.push(calc_observation(
            quantum_state,
            hamiltonian,
            pos.clone(),
            true,
        ));
    }
    Tensor::stack(&vals, 0)
}

pub fn bipartite_entanglement_entropy(
    quantum_state: &Tensor,
    qubit_idx: Option<Vec<i64>>,
) -> Tensor {
    let eps = 1e-14;
    check_state_tensor(quantum_state).expect("invalid state");
    let num_qubits = quantum_state.dim() as i64;
    let indices: Vec<i64> = match qubit_idx {
        None => (1..num_qubits).collect(),
        Some(v) => v,
    };
    assert!(indices.iter().all(|&i| i >= 1 && i <= num_qubits));
    let mut ent: Vec<Tensor> = Vec::with_capacity(indices.len());
    for idx in indices {
        let left = (0..idx).collect::<Vec<_>>();
        let right = (idx..num_qubits).collect::<Vec<_>>();
        // reshape to [prod(left), prod(right)]
        let dl = 1_i64 << (left.len() as i64);
        let dr = 1_i64 << (right.len() as i64);
        let m = quantum_state.view([dl, dr]);
        // singular values via SVD
        let (_u, s, _v) = m.svd(true, true);
        let s2 = &s * &s;
        let e = -(&s2 * (&s2 + eps).log()).sum(s.kind());
        ent.push(e);
    }
    Tensor::stack(&ent, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, IndexOp, Kind};

    #[test]
    fn test_rdm_single_qubit() {
        // |00>
        let s = {
            let mut v = Tensor::zeros([4], (Kind::ComplexDouble, Device::Cpu));
            let one = Tensor::from(1.0).to_kind(Kind::ComplexDouble);
            v = v
                .f_index_put_(&[Some(Tensor::from(0))], &one, false)
                .unwrap();
            v.view([2, 2])
        };
        let rdm0 = calc_reduced_density_matrix(&s, vec![0]);
        let rdm1 = calc_reduced_density_matrix(&s, vec![1]);
        assert!(rdm0.i((0, 0)).abs().double_value(&[]) > 0.9);
        assert!(rdm1.i((0, 0)).abs().double_value(&[]) > 0.9);
    }
}
