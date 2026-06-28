//! Functional MPS helpers.

use tch::{Device, IndexOp, Kind, Tensor};

use crate::types::{MPSType, OrthogonalizationMode};
use crate::utils::checking::check_state_tensor;
use crate::utils::devices::linalg_work_device;
use crate::utils::tensors::tensor_contract;

/// Generate random local tensors for an MPS.
pub fn gen_random_mps_tensors(
    length: i64,
    physical_dim: i64,
    virtual_dim: i64,
    mps_type: MPSType,
    kind: Kind,
    device: Device,
) -> Vec<Tensor> {
    assert!(length > 0, "length must be positive");
    assert!(physical_dim > 0, "physical_dim must be positive");
    assert!(virtual_dim > 0, "virtual_dim must be positive");
    match mps_type {
        MPSType::Open => {
            if length == 1 {
                return vec![Tensor::randn([1, physical_dim, 1], (kind, device))];
            }
            let mut tensors = Vec::with_capacity(length as usize);
            tensors.push(Tensor::randn(
                [1, physical_dim, virtual_dim],
                (kind, device),
            ));
            for _ in 0..length - 2 {
                tensors.push(Tensor::randn(
                    [virtual_dim, physical_dim, virtual_dim],
                    (kind, device),
                ));
            }
            tensors.push(Tensor::randn(
                [virtual_dim, physical_dim, 1],
                (kind, device),
            ));
            tensors
        }
        MPSType::Periodic => (0..length)
            .map(|_| Tensor::randn([virtual_dim, physical_dim, virtual_dim], (kind, device)))
            .collect(),
    }
}

/// Calculate the global tensor by a single explicit contraction expression.
pub fn calc_global_tensor_by_contract(mps_tensors: &[Tensor]) -> Tensor {
    assert!(!mps_tensors.is_empty(), "MPS must have at least one tensor");
    let length = mps_tensors.len();
    let mps_type = MPSType::from_tensors(mps_tensors);

    if length == 1 {
        return match mps_type {
            MPSType::Open => mps_tensors[0].squeeze(),
            MPSType::Periodic => {
                let equation = einops::einsum_str("left physical left -> physical");
                Tensor::einsum(&equation, &[&mps_tensors[0]], None::<i64>).squeeze()
            }
        };
    }

    let mut input_terms = Vec::with_capacity(length);
    let mut shared_groups = Vec::with_capacity(match mps_type {
        MPSType::Open => length - 1,
        MPSType::Periodic => length,
    });
    let mut physical_labels = Vec::with_capacity(length);
    for idx in 0..length {
        input_terms.push(format!("left_{idx} physical_{idx} right_{idx}"));
        physical_labels.push(format!("physical_{idx}"));
        if idx + 1 < length {
            shared_groups.push(vec![format!("right_{idx}"), format!("left_{}", idx + 1)]);
        }
    }
    let output_labels = match mps_type {
        MPSType::Open => {
            let mut labels = Vec::with_capacity(length + 2);
            labels.push("left_0".to_string());
            labels.extend(physical_labels);
            labels.push(format!("right_{}", length - 1));
            labels
        }
        MPSType::Periodic => {
            shared_groups.push(vec![format!("right_{}", length - 1), "left_0".to_string()]);
            physical_labels
        }
    };
    let equation = format!("{} -> {}", input_terms.join(", "), output_labels.join(" "));
    let tensor_refs = mps_tensors.iter().collect::<Vec<_>>();
    tensor_contract(&tensor_refs, &equation, shared_groups).squeeze()
}

/// Calculate the global tensor by sequential tensordot.
pub fn calc_global_tensor_by_tensordot(mps_tensors: &[Tensor]) -> Tensor {
    assert!(!mps_tensors.is_empty(), "MPS must have at least one tensor");
    let mut psi = mps_tensors[0].shallow_clone();
    for tensor in mps_tensors.iter().skip(1) {
        let dim = psi.dim() as i64 - 1;
        psi = psi.tensordot(tensor, [dim], [0]);
    }
    match MPSType::from_tensors(mps_tensors) {
        MPSType::Open => psi.squeeze(),
        MPSType::Periodic => {
            let dim = psi.dim() as i64;
            psi.diagonal(0, 0, dim - 1)
                .sum_dim_intlist([-1].as_slice(), false, None::<Kind>)
        }
    }
}

/// Calculate the MPS norm factors.
pub fn calculate_mps_norm_factors(mps_tensors: &[Tensor], efficient_mode: bool) -> Tensor {
    assert!(!mps_tensors.is_empty(), "MPS must have at least one tensor");
    let conjugates: Vec<Tensor> = mps_tensors.iter().map(Tensor::conj).collect();
    let length = mps_tensors.len();
    let device = conjugates[0].device();
    let kind = conjugates[0].kind();
    match MPSType::from_tensors(mps_tensors) {
        MPSType::Open => {
            let mut v = Tensor::ones([1, 1], (kind, device));
            let mut factors = Vec::with_capacity(length);
            for i in 0..length {
                if efficient_mode {
                    v = Tensor::einsum("ab,aix->bix", &[&v, &conjugates[i]], None::<i64>);
                    v = Tensor::einsum("bix,biy->xy", &[&v, &mps_tensors[i]], None::<i64>);
                } else {
                    v = Tensor::einsum(
                        "ab,aix,biy->xy",
                        &[&v, &conjugates[i], &mps_tensors[i]],
                        None::<i64>,
                    );
                }
                let norm_factor = v.norm();
                v = &v / &norm_factor;
                factors.push(norm_factor);
            }
            Tensor::stack(&factors, 0)
        }
        MPSType::Periodic => {
            let virtual_dim = mps_tensors[0].size()[0];
            let mut factors = Vec::with_capacity(length);
            let mut v = Tensor::eye(virtual_dim * virtual_dim, (kind, device)).reshape([
                virtual_dim,
                virtual_dim,
                virtual_dim,
                virtual_dim,
            ]);
            for i in 0..length {
                v = Tensor::einsum(
                    "uvap,adb,pdq->uvbq",
                    &[&v, &conjugates[i], &mps_tensors[i]],
                    None::<i64>,
                );
                let norm_factor = v.norm();
                v = &v / &norm_factor;
                factors.push(norm_factor);
            }
            let final_factor = Tensor::einsum("acac->", &[&v], None::<i64>);
            let last = factors.pop().expect("length checked");
            factors.push(last * final_factor);
            Tensor::stack(&factors, 0)
        }
    }
}

/// Normalize MPS tensors out of place.
pub fn normalize_mps(mps_tensors: &[Tensor]) -> Vec<Tensor> {
    assert!(!mps_tensors.is_empty(), "MPS must have at least one tensor");
    let factors = calculate_mps_norm_factors(mps_tensors, true)
        .sqrt()
        .reciprocal();
    mps_tensors
        .iter()
        .enumerate()
        .map(|(idx, tensor)| tensor * factors.i(idx as i64))
        .collect()
}

/// Calculate the inner-product factors between two MPS.
pub fn calc_inner_product(mps0: &[Tensor], mps1: &[Tensor]) -> Tensor {
    assert_eq!(mps0.len(), mps1.len(), "length of two MPS must be the same");
    assert_eq!(mps0[0].size()[0], mps0[mps0.len() - 1].size()[2]);
    assert_eq!(mps1[0].size()[0], mps1[mps1.len() - 1].size()[2]);
    assert_eq!(mps0[0].kind(), mps1[0].kind());
    assert_eq!(mps0[0].device(), mps1[0].device());
    let endpoint0 = mps0[0].size()[0];
    let endpoint1 = mps1[0].size()[0];
    let kind = mps0[0].kind();
    let device = mps0[0].device();
    let v0 = Tensor::eye(endpoint0, (kind, device));
    let v1 = Tensor::eye(endpoint1, (kind, device));
    let mut v = Tensor::einsum("ab,xy->axby", &[&v0, &v1], None::<i64>);
    let mut factors = Vec::with_capacity(mps0.len() + 1);
    for i in 0..mps0.len() {
        v = Tensor::einsum(
            "uvap,adb,pdq->uvbq",
            &[&v, &mps0[i].conj(), &mps1[i]],
            None::<i64>,
        );
        let product_factor = v.norm();
        v = &v / &product_factor;
        factors.push(product_factor);
    }
    if v.numel() > 1 {
        factors.push(Tensor::einsum("acac->", &[&v], None::<i64>));
    } else {
        factors.push(v.reshape([]));
    }
    Tensor::stack(&factors, 0)
}

/// One left-to-right orthogonalization step.
pub fn orthogonalize_left2right_step(
    mps_tensors: &[Tensor],
    local_tensor_idx: usize,
    mode: OrthogonalizationMode,
    truncate_dim: Option<i64>,
    normalize: bool,
    check_nan: bool,
) -> (Tensor, Tensor) {
    let length = mps_tensors.len();
    assert!(length > 1, "mps_tensors must have at least 2 tensors");
    assert!(
        local_tensor_idx < length - 1,
        "local_tensor_idx must be in [0, length - 2]"
    );
    let local_tensor = &mps_tensors[local_tensor_idx];
    let shape = local_tensor.size();
    assert_eq!(shape.len(), 3, "MPS local tensor must be rank-3");
    let mut truncate = None;
    if let Some(dim) = truncate_dim {
        assert!(dim > 0, "truncate_dim must be positive");
        assert_eq!(
            mode,
            OrthogonalizationMode::Svd,
            "mode must be 'svd' when truncate_dim is provided"
        );
        truncate = Some(dim.min(shape[2]));
    }
    let view_matrix = local_tensor.view([-1, shape[2]]);
    let original_device = view_matrix.device();
    let work_device = linalg_work_device(original_device);
    let view_matrix_work = view_matrix.to_device(work_device);
    let (u, mut r) = match mode {
        OrthogonalizationMode::Svd => {
            let (mut u, mut s, mut vh) = Tensor::linalg_svd(&view_matrix_work, false, "");
            if let Some(dim) = truncate {
                u = u.i((.., ..dim));
                s = s.i(..dim);
                vh = vh.i((..dim, ..));
            }
            (
                u.to_device(original_device),
                (s.unsqueeze(1) * vh).to_device(original_device),
            )
        }
        OrthogonalizationMode::Qr => {
            let (q, r) = Tensor::linalg_qr(&view_matrix_work, "reduced");
            (q.to_device(original_device), r.to_device(original_device))
        }
    };
    if normalize {
        r = &r / r.norm();
    }
    let new_local_tensor = u.reshape([shape[0], shape[1], -1]);
    let new_local_tensor_right = Tensor::einsum(
        "ab,bcd->acd",
        &[&r, &mps_tensors[local_tensor_idx + 1]],
        None::<i64>,
    );
    if check_nan {
        assert!(
            new_local_tensor.isnan().any().int64_value(&[]) == 0,
            "Due to numerical errors, the new local tensor may contain nan values."
        );
        assert!(
            new_local_tensor_right.isnan().any().int64_value(&[]) == 0,
            "Due to numerical errors, the new local tensor right may contain nan values."
        );
    }
    (new_local_tensor, new_local_tensor_right)
}

/// One right-to-left orthogonalization step.
pub fn orthogonalize_right2left_step(
    mps_tensors: &[Tensor],
    local_tensor_idx: usize,
    mode: OrthogonalizationMode,
    truncate_dim: Option<i64>,
    normalize: bool,
    check_nan: bool,
) -> (Tensor, Tensor) {
    let length = mps_tensors.len();
    assert!(length > 1, "mps_tensors must have at least 2 tensors");
    assert!(
        (1..length).contains(&local_tensor_idx),
        "local_tensor_idx must be in [1, length - 1]"
    );
    let local_tensor = &mps_tensors[local_tensor_idx];
    let shape = local_tensor.size();
    assert_eq!(shape.len(), 3, "MPS local tensor must be rank-3");
    let mut truncate = None;
    if let Some(dim) = truncate_dim {
        assert!(dim > 0, "truncate_dim must be positive");
        assert_eq!(
            mode,
            OrthogonalizationMode::Svd,
            "mode must be 'svd' when truncate_dim is provided"
        );
        truncate = Some(dim.min(shape[0]));
    }
    let view_matrix = local_tensor.view([shape[0], -1]).transpose(0, 1);
    let original_device = view_matrix.device();
    let work_device = linalg_work_device(original_device);
    let view_matrix_work = view_matrix.to_device(work_device);
    let (u, mut r) = match mode {
        OrthogonalizationMode::Svd => {
            let (mut u, mut s, mut vh) = Tensor::linalg_svd(&view_matrix_work, false, "");
            if let Some(dim) = truncate {
                u = u.i((.., ..dim));
                s = s.i(..dim);
                vh = vh.i((..dim, ..));
            }
            (
                u.to_device(original_device),
                (s.unsqueeze(1) * vh).to_device(original_device),
            )
        }
        OrthogonalizationMode::Qr => {
            let (q, r) = Tensor::linalg_qr(&view_matrix_work, "reduced");
            (q.to_device(original_device), r.to_device(original_device))
        }
    };
    if normalize {
        r = &r / r.norm();
    }
    let new_local_tensor = u.transpose(0, 1).reshape([-1, shape[1], shape[2]]);
    let new_local_tensor_left = Tensor::einsum(
        "abc,dc->abd",
        &[&mps_tensors[local_tensor_idx - 1], &r],
        None::<i64>,
    );
    if check_nan {
        assert!(
            new_local_tensor_left.isnan().any().int64_value(&[]) == 0,
            "Due to numerical errors, the new local tensor left may contain nan values."
        );
        assert!(
            new_local_tensor.isnan().any().int64_value(&[]) == 0,
            "Due to numerical errors, the new local tensor may contain nan values."
        );
    }
    (new_local_tensor_left, new_local_tensor)
}

/// Orthogonalize tensors between two positions and return changed indices.
pub fn orthogonalize_arange(
    mps_tensors: &[Tensor],
    start_idx: usize,
    end_idx: usize,
    mode: OrthogonalizationMode,
    truncate_dim: Option<i64>,
    normalize: bool,
    check_nan: bool,
) -> (Vec<Tensor>, Vec<usize>) {
    let length = mps_tensors.len();
    assert!(length > 1, "mps_tensors must have at least 2 tensors");
    assert!(
        start_idx < length && end_idx < length,
        "start_idx and end_idx must be in [0, length - 1]"
    );
    let mut tensors: Vec<Tensor> = mps_tensors.iter().map(Tensor::shallow_clone).collect();
    let mut changed = Vec::<usize>::new();
    if start_idx < end_idx {
        for idx in start_idx..end_idx {
            let (local, right) = orthogonalize_left2right_step(
                &tensors,
                idx,
                mode,
                truncate_dim,
                normalize,
                check_nan,
            );
            tensors[idx] = local;
            tensors[idx + 1] = right;
            push_unique(&mut changed, idx);
            push_unique(&mut changed, idx + 1);
        }
    } else if start_idx > end_idx {
        for idx in ((end_idx + 1)..=start_idx).rev() {
            let (left, local) = orthogonalize_right2left_step(
                &tensors,
                idx,
                mode,
                truncate_dim,
                normalize,
                check_nan,
            );
            tensors[idx - 1] = left;
            tensors[idx] = local;
            push_unique(&mut changed, idx - 1);
            push_unique(&mut changed, idx);
        }
    }
    changed.sort_unstable();
    (tensors, changed)
}

/// Tensor-train decomposition of a quantum state tensor.
pub fn tt_decomposition(
    state_tensor: &Tensor,
    max_rank: Option<i64>,
    use_svd: bool,
) -> (Vec<Tensor>, Vec<i64>) {
    check_state_tensor(state_tensor);
    if let Some(max_rank) = max_rank {
        assert!(max_rank > 0, "max_rank must be greater than 0");
    }
    let use_svd = use_svd || max_rank.is_some();
    let physical_dim = state_tensor.size()[0];
    let shape = state_tensor.size();
    let n_qubits = state_tensor.dim();
    let mut left_dim = 1;
    let mut local_tensors = Vec::with_capacity(n_qubits);
    let mut remained_tensor = state_tensor.shallow_clone();
    let mut clipped_ranks = Vec::new();
    for &mid_dim in shape.iter().take(n_qubits - 1) {
        if use_svd {
            let work_device = linalg_work_device(remained_tensor.device());
            let matrix = remained_tensor.reshape([left_dim * mid_dim, -1]);
            let original_device = matrix.device();
            let (mut q, mut s, mut vh) =
                Tensor::linalg_svd(&matrix.to_device(work_device), false, "");
            q = q.to_device(original_device);
            s = s.to_device(original_device);
            vh = vh.to_device(original_device);
            let rank = max_rank.map_or_else(|| s.size()[0], |m| m.min(s.size()[0]));
            q = q.i((.., ..rank));
            s = s.i(..rank);
            vh = vh.i((..rank, ..));
            remained_tensor = s.unsqueeze(1) * vh;
            local_tensors.push(q.view([left_dim, mid_dim, rank]));
            left_dim = rank;
            clipped_ranks.push(rank);
        } else {
            let work_device = linalg_work_device(remained_tensor.device());
            let matrix = remained_tensor.reshape([left_dim * mid_dim, -1]);
            let original_device = matrix.device();
            let (q, r) = Tensor::linalg_qr(&matrix.to_device(work_device), "reduced");
            let q = q.to_device(original_device);
            remained_tensor = r.to_device(original_device);
            let new_left_dim = remained_tensor.size()[0];
            local_tensors.push(q.view([left_dim, mid_dim, new_left_dim]));
            left_dim = new_left_dim;
        }
    }
    local_tensors.push(remained_tensor.view([left_dim, physical_dim, 1]));
    (local_tensors, clipped_ranks)
}

/// Project multiple qubits and return local tensors for the projected MPS.
pub fn project_multi_qubits(
    mps_local_tensors: &[Tensor],
    qubit_indices: &[usize],
    project_to_states: ProjectToStates<'_>,
) -> Vec<Tensor> {
    assert_eq!(
        qubit_indices.len(),
        project_to_states.len(),
        "project_to_states must match qubit_indices"
    );
    let mut local_tensors: Vec<Tensor> = mps_local_tensors
        .iter()
        .map(Tensor::shallow_clone)
        .collect();
    if qubit_indices.is_empty() {
        return local_tensors;
    }
    match project_to_states {
        ProjectToStates::Vectors(states) => {
            assert_eq!(states.dim(), 2, "states must be a 2D tensor");
            for (row_idx, &qubit_idx) in qubit_indices.iter().enumerate() {
                let local = &local_tensors[qubit_idx];
                let state = states.i(row_idx as i64);
                assert_eq!(
                    local.size()[1],
                    state.size()[0],
                    "The feature dimension of the project_to_states must match the physical dimension of the local tensor of MPS"
                );
                local_tensors[qubit_idx] =
                    Tensor::einsum("lpr,p->lr", &[local, &state], None::<i64>);
            }
        }
        ProjectToStates::Indices(states) => {
            for (idx, &qubit_idx) in qubit_indices.iter().enumerate() {
                let local = &local_tensors[qubit_idx];
                let state_idx = states[idx];
                assert!(
                    0 <= state_idx && state_idx < local.size()[1],
                    "state_idx must be in range"
                );
                local_tensors[qubit_idx] = local.i((.., state_idx, ..));
            }
        }
    }
    let mut sorted = qubit_indices.to_vec();
    sorted.sort_unstable_by(|a, b| b.cmp(a));
    for &qubit_idx in sorted.iter().take(sorted.len().saturating_sub(1)) {
        assert!(qubit_idx > 0);
        let left = qubit_idx - 1;
        local_tensors[left] = local_tensors[left].tensordot(&local_tensors[qubit_idx], [-1], [0]);
        let _ = local_tensors.remove(qubit_idx);
    }
    if local_tensors.len() > 1 {
        let qubit_idx = *sorted.last().expect("non-empty");
        if qubit_idx == 0 {
            local_tensors[1] = local_tensors[0].tensordot(&local_tensors[1], [-1], [0]);
        } else {
            let left = qubit_idx - 1;
            local_tensors[left] =
                local_tensors[left].tensordot(&local_tensors[qubit_idx], [-1], [0]);
        }
        let _ = local_tensors.remove(qubit_idx);
    }
    for tensor in &mut local_tensors {
        if tensor.dim() == 2 {
            *tensor = tensor.unsqueeze(1);
        } else {
            assert_eq!(tensor.dim(), 3, "Unexpected tensor dimension: bug?");
        }
    }
    local_tensors
}

/// Projection target for one or more physical sites.
pub enum ProjectToStates<'a> {
    /// Vector states with shape `(project_qubit_num, physical_dim)`.
    Vectors(&'a Tensor),
    /// Basis-state indices.
    Indices(&'a [i64]),
}

impl ProjectToStates<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Vectors(tensor) => tensor.size()[0] as usize,
            Self::Indices(indices) => indices.len(),
        }
    }
}

fn push_unique(values: &mut Vec<usize>, value: usize) {
    if !values.contains(&value) {
        values.push(value);
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind};

    use super::*;

    #[test]
    fn global_tensor_contract_matches_tensordot_for_open_and_periodic_mps() {
        for mps_type in [MPSType::Open, MPSType::Periodic] {
            let tensors = gen_random_mps_tensors(4, 2, 3, mps_type, Kind::Float, Device::Cpu);
            let contract = calc_global_tensor_by_contract(&tensors);
            let tensordot = calc_global_tensor_by_tensordot(&tensors);
            assert!(contract.allclose(&tensordot, 1e-5, 1e-6, false));
        }
    }
}
