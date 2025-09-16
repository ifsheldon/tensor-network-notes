use crate::mps::modules::MPS;
use crate::tensor_gates::functional::apply_gate;
// use crate::utils::mapping::view_gate_matrix_as_tensor; // not required in current implementation
use crate::constants::NO_OPT_PATH;
use crate::mps::functional::orthogonalize_arange;
use crate::types::*;
use crate::utils::checking::check_quantum_gate;
use tch::Tensor;

/// Minimal TEBD-like evolution by repeatedly applying a two-site gate at
/// specified positions, with state renormalization after each step.
pub fn time_evolving_block_decimation(
    initial_state: &Tensor,
    two_site_gate: &Tensor,
    positions: &[Vec<UIdx>],
    steps: Num,
) -> Tensor {
    let mut state = initial_state.shallow_clone();
    for _ in 0..steps {
        for pos in positions {
            state = apply_gate(&state, two_site_gate, pos.clone(), None);
        }
        state = &state / state.norm();
    }
    state
}

fn two_site_gate_from_h(hamiltonian: &Tensor, tau: f64) -> Tensor {
    // Accept [4,4] or [2,2,2,2]; return matrix [4,4]
    let num_q = check_quantum_gate(hamiltonian, None, true).unwrap();
    assert_eq!(num_q, 2, "Only 2-body Hamiltonians supported");
    let mat = if hamiltonian.dim() == 2 {
        hamiltonian.shallow_clone()
    } else {
        hamiltonian.view([4, 4])
    };
    (-(tau) * &mat).matrix_exp()
}

fn decompose_two_site_gate(gate_mat: &Tensor) -> (Tensor, Tensor) {
    // gate_mat is [4,4] mapping (i',j') x (i,j). Decompose into sum_g A_g(i',i) B_g(j',j).
    let k = gate_mat.kind();
    let dev = gate_mat.device();
    let gate_tt = gate_mat.view([2, 2, 2, 2]); // [i', j', i, j]
    // Arrange as [i', i, j', j] then flatten to [4, 4]
    let left_right = gate_tt.permute([0, 2, 1, 3]).contiguous().view([4, 4]);
    let (u, s, v) = left_right.svd(false, true);
    let r = s.size()[0];
    let sqrt_s = s.sqrt();
    let diag = Tensor::diag_embed(&sqrt_s, 0, -2, -1); // [r,r]
    let gl = u.matmul(&diag).view([2, 2, r]).permute([0, 2, 1]); // [i', g, i]
    let gr = diag.matmul(&v).view([r, 2, 2]).permute([1, 0, 2]); // [j', g, j]
    (gl.to_kind(k).to_device(dev), gr.to_kind(k).to_device(dev))
}

// Removed adjacent/Swap-based helpers; we use MPO-style factorization below.

fn apply_two_site_gate_long_range(
    mps: &mut MPS,
    gate_mat: &Tensor, // [4,4]
    p0: UIdx,
    p1: UIdx,
    max_virtual_dim: Num,
) {
    assert!(p0 < p1 && p1 < mps.len());
    let p0: usize = p0.cast();
    let p1: usize = p1.cast();
    // Decompose gate into left/right factors with auxiliary bond g
    let (gl, gr) = decompose_two_site_gate(gate_mat); // gl: [i',g,i], gr: [j',g,j]
    let g_dim = gl.size()[1];
    let mut locals: Vec<Tensor> = mps
        .local_tensors()
        .iter()
        .map(|t| t.shallow_clone())
        .collect();
    // Left site update at p0: [l, p, r] x [p', g, p] -> [l, p', g, r] -> [l, p', (g r)]
    {
        let lt = &locals[p0];
        let new_l = Tensor::einsum(
            "l p r, p2 g p -> l p2 g r",
            &[lt.shallow_clone(), gl.shallow_clone()],
            NO_OPT_PATH,
        );
        let l = new_l.size()[0];
        let p2 = new_l.size()[1];
        let r = new_l.size()[3];
        locals[p0] = new_l.view([l, p2, g_dim * r]);
    }
    // Middle sites p0+1..p1-1: insert identity on g
    if p1 > p0 + 1 {
        let eye = Tensor::eye(g_dim, (locals[p0].kind(), locals[p0].device()));
        for t in locals.iter_mut().take(p1).skip(p0 + 1) {
            let new_t = Tensor::einsum(
                "g0 g1, l p r -> g0 l p g1 r",
                &[eye.shallow_clone(), t.shallow_clone()],
                NO_OPT_PATH,
            );
            let l = new_t.size()[0] * new_t.size()[1];
            let p = new_t.size()[2];
            let r = new_t.size()[3] * new_t.size()[4];
            *t = new_t.view([l, p, r]);
        }
    }
    // Right site update at p1: [l, p, r] x [p2, g, p] -> [g, l, p2, r] -> [(g l), p2, r]
    {
        let rt = &locals[p1];
        let new_r = Tensor::einsum(
            "l p r, p2 g p -> g l p2 r",
            &[rt.shallow_clone(), gr.shallow_clone()],
            NO_OPT_PATH,
        );
        let g = new_r.size()[0];
        let l = new_r.size()[1];
        let p2 = new_r.size()[2];
        let r = new_r.size()[3];
        locals[p1] = new_r.view([g * l, p2, r]);
    }
    // Compress along [p0..p1] by sweeping with SVD truncation
    let (mps2, _) = orthogonalize_arange(
        &locals,
        p0.cast(),
        p1.cast(),
        "svd",
        Some(max_virtual_dim),
        false,
        false,
        true,
    );
    let (mps3, _) = orthogonalize_arange(
        &mps2,
        p1.cast(),
        p0.cast(),
        "svd",
        Some(max_virtual_dim),
        false,
        false,
        true,
    );
    for (idx, t) in mps3.iter().enumerate().take(p1 + 1).skip(p0) {
        mps.force_set_local_tensor(idx, t.shallow_clone());
    }
}

/// Compute local bond energies for nearest-neighbor positions using two-body
/// reduced density matrices and `[4,4]` Hamiltonians.
pub fn calculate_mps_local_energies(
    mps: &mut MPS,
    hamiltonians: &[Tensor],
    positions: &[(UIdx, UIdx)],
) -> Tensor {
    let mut out: Vec<Tensor> = Vec::with_capacity(positions.len());
    for (idx, pos) in positions.iter().enumerate() {
        let (i, j) = *pos;
        assert_eq!(j, i + 1, "Only nearest-neighbor is supported in Rust TEBD");
        let rdm = mps.two_body_reduced_density_matrix(i, j, true); // [4,4]
        let h = if hamiltonians.len() == 1 {
            hamiltonians[0].shallow_clone()
        } else {
            hamiltonians[idx].shallow_clone()
        };
        let h_mat = if h.dim() == 2 { h } else { h.view([4, 4]) };
        let e = (h_mat.transpose(0, 1) * &rdm).sum(h_mat.kind());
        out.push(e);
    }
    Tensor::stack(&out, 0)
}

/// A more complete TEBD evolution with nearest-neighbor two-site gates.
#[allow(clippy::too_many_arguments)]
pub fn tebd(
    hamiltonians: &[Tensor],    // either len=1 or len==positions.len()
    positions: &[(UIdx, UIdx)], // bonds; only nearest-neighbor supported
    mut mps: MPS,
    mut tau: f64,
    iterations: Num,
    calc_observation_iters: Num,
    e0_eps: f64,
    tau_min: f64,
    least_iters_for_tau: Num,
    max_virtual_dim: Num,
) -> (MPS, Tensor) {
    assert!(iterations > 0 && calc_observation_iters > 0);
    assert!(e0_eps > 0.0 && tau > tau_min && tau_min > 0.0);
    let mut obs: Vec<Tensor> = Vec::new();
    let mut last_e = Tensor::from(1.0);
    let h_is_single = hamiltonians.len() == 1;
    let mut step_count_since_tau = 0;
    for it in 0..iterations {
        for (k, pos) in positions.iter().enumerate() {
            let (i, j) = *pos;
            let h = if h_is_single {
                &hamiltonians[0]
            } else {
                &hamiltonians[k]
            };
            let gate = two_site_gate_from_h(h, tau);
            apply_two_site_gate_long_range(&mut mps, &gate, i, j, max_virtual_dim);
        }
        mps.center_normalize();
        step_count_since_tau += 1;

        if it % calc_observation_iters == 0 {
            let en_local = calculate_mps_local_energies(&mut mps, hamiltonians, positions);
            let e = en_local.sum(en_local.kind());
            obs.push(e.shallow_clone());
            let diff = (&e - &last_e).abs();
            if diff.double_value(&[]) < e0_eps * tau && step_count_since_tau >= least_iters_for_tau
            {
                tau *= 0.5;
                step_count_since_tau = 0;
            }
            if tau < tau_min {
                break;
            }
            last_e = e;
        }
    }
    (mps, Tensor::stack(&obs, 0))
}
