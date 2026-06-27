//! Time-evolving block decimation helpers.

use tch::{Kind, Tensor};

use crate::mps::MPS;
use crate::types::OrthogonalizationMode;
use crate::utils::checking::check_quantum_gate;
use crate::utils::mapping::view_gate_tensor_as_matrix;

/// Apply two local gate factors to an MPS tensor list.
pub fn evolve_gate_2body(
    mps_local_tensors: &[Tensor],
    gl: &Tensor,
    gr: &Tensor,
    p0: usize,
    p1: usize,
) -> Vec<Tensor> {
    assert!(p0 < p1);
    assert_eq!(gl.size()[1], gr.size()[1]);
    let g_dim = gl.size()[1];
    let mut local_tensors: Vec<Tensor> = mps_local_tensors
        .iter()
        .map(Tensor::shallow_clone)
        .collect();
    let left = Tensor::einsum("lcr,agc->lagr", &[&local_tensors[p0], gl], None::<i64>);
    local_tensors[p0] = left.reshape([left.size()[0], left.size()[1], g_dim * left.size()[3]]);
    let right = Tensor::einsum("lpr,bgp->glbr", &[&local_tensors[p1], gr], None::<i64>);
    local_tensors[p1] = right.reshape([g_dim * right.size()[1], right.size()[2], right.size()[3]]);
    let eye = Tensor::eye(g_dim, (Kind::Int, mps_local_tensors[0].device()));
    for tensor in local_tensors.iter_mut().take(p1).skip(p0 + 1) {
        let current = tensor.shallow_clone();
        let expanded = Tensor::einsum("ab,lpr->al pbr", &[&eye, &current], None::<i64>);
        let size = expanded.size();
        *tensor = expanded.reshape([size[0] * size[1], size[2], size[3] * size[4]]);
    }
    local_tensors
}

/// Direction to move center between two interactions.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CenterDirection {
    /// Move right to left.
    RightToLeft,
    /// Move left to right.
    LeftToRight,
}

/// Decide the next center movement direction.
pub fn direction_to_next_center(l0: i64, r0: i64, l1: i64, r1: i64) -> CenterDirection {
    let l_min = (l0 - l1).abs().min((l0 - r1).abs());
    let r_min = (r0 - l1).abs().min((r0 - r1).abs());
    if l_min < r_min {
        CenterDirection::RightToLeft
    } else {
        CenterDirection::LeftToRight
    }
}

/// Calculate local two-body energies.
pub fn calculate_mps_local_energies(
    mps: &mut MPS,
    hamiltonians: &[Tensor],
    positions: &[Vec<usize>],
) -> Tensor {
    let hs: Vec<Tensor> = if hamiltonians.len() == 1 {
        (0..positions.len())
            .map(|_| hamiltonians[0].shallow_clone())
            .collect()
    } else {
        assert_eq!(hamiltonians.len(), positions.len());
        hamiltonians.iter().map(Tensor::shallow_clone).collect()
    };
    let mut energies = Vec::new();
    for (pos, hamiltonian) in positions.iter().zip(hs.iter()) {
        assert_eq!(pos.len(), 2, "Only support 2-body interaction for now");
        let rdm = mps.two_body_reduced_density_matrix(pos[0], pos[1], true);
        let h = view_gate_tensor_as_matrix(&hamiltonian.to_device(rdm.device()), None);
        energies.push(Tensor::einsum("ab,ba->", &[&h, &rdm], None::<i64>));
    }
    Tensor::stack(&energies, 0)
}

/// Minimal TEBD loop for two-body Hamiltonians.
pub fn tebd(
    hamiltonians: &[Tensor],
    positions: &[Vec<usize>],
    mut mps: MPS,
    mut tau: f64,
    iterations: i64,
    calc_observation_iters: i64,
    e0_eps: f64,
    tau_min: f64,
    max_virtual_dim: i64,
) -> (MPS, Tensor) {
    assert!(1.0 > tau && tau >= 0.0 && 1.0 > tau_min && tau_min >= 0.0);
    assert!(iterations >= 0 && calc_observation_iters >= 0);
    assert!(max_virtual_dim >= 1);
    assert!(!positions.is_empty());
    for h in hamiltonians {
        check_quantum_gate(h, Some(2), false);
    }
    mps.center_orthogonalize(
        positions[0][1] as isize,
        OrthogonalizationMode::Svd,
        Some(max_virtual_dim),
        true,
        false,
    );
    mps.normalize(false);
    let mut local_energies = Tensor::zeros([positions.len() as i64], (mps.kind(), mps.device()));
    for t in 0..iterations {
        for (p, pos) in positions.iter().enumerate() {
            let p_left = pos[0];
            let p_right = pos[1];
            if (mps.center().expect("center") as isize - p_left as isize).abs()
                < (mps.center().expect("center") as isize - p_right as isize).abs()
            {
                mps.center_orthogonalize(
                    p_left as isize,
                    OrthogonalizationMode::Qr,
                    None,
                    true,
                    false,
                );
            } else {
                mps.center_orthogonalize(
                    p_right as isize,
                    OrthogonalizationMode::Qr,
                    None,
                    true,
                    false,
                );
            }
            let h = if hamiltonians.len() == 1 {
                &hamiltonians[0]
            } else {
                &hamiltonians[p]
            };
            let gate = (-tau * view_gate_tensor_as_matrix(h, None))
                .matrix_exp()
                .reshape([2, 2, 2, 2]);
            let gl = gate.permute([0, 1, 3, 2]).reshape([2, 4, 2]);
            let eye = Tensor::eye(mps.physical_dim(), (mps.kind(), mps.device()));
            let gr = Tensor::einsum("ab,cd->acbd", &[&eye, &eye], None::<i64>).reshape([2, 4, 2]);
            let evolved = evolve_gate_2body(&mps.local_tensors(), &gl, &gr, p_left, p_right);
            mps = MPS::from_tensors(evolved);
            mps.center_orthogonalize(
                p_left as isize,
                OrthogonalizationMode::Svd,
                Some(max_virtual_dim),
                true,
                false,
            );
            mps.center_normalize();
        }
        if calc_observation_iters == 0
            || (t + 1) % calc_observation_iters == 0
            || t == iterations - 1
        {
            let new_energies = calculate_mps_local_energies(&mut mps, hamiltonians, positions);
            let avg_diff = (&new_energies - &local_energies).abs().mean(None::<Kind>);
            local_energies = new_energies;
            if avg_diff.double_value(&[]) < e0_eps || t == iterations - 1 {
                tau *= 0.5;
                if tau < tau_min || t == iterations - 1 {
                    break;
                }
            }
        }
    }
    (mps, local_energies)
}
