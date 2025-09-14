use crate::utils::einsum::named_einsum;
use tch::{Device, IndexOp, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MPSType {
    Open,
    Periodic,
}

pub fn get_mps_type(mps: &[Tensor]) -> MPSType {
    if mps.first().unwrap().size()[0] == 1 && mps.last().unwrap().size()[2] == 1 {
        MPSType::Open
    } else {
        MPSType::Periodic
    }
}

pub fn gen_random_mps_tensors(
    length: i64,
    physical_dim: i64,
    virtual_dim: i64,
    mps_type: MPSType,
    kind: Kind,
    device: Device,
) -> Vec<Tensor> {
    match mps_type {
        MPSType::Open => {
            let mut out = Vec::with_capacity(length as usize);
            out.push(Tensor::randn(
                [1, physical_dim, virtual_dim],
                (kind, device),
            ));
            for _ in 0..(length - 2) {
                out.push(Tensor::randn(
                    [virtual_dim, physical_dim, virtual_dim],
                    (kind, device),
                ));
            }
            out.push(Tensor::randn(
                [virtual_dim, physical_dim, 1],
                (kind, device),
            ));
            out
        }
        MPSType::Periodic => {
            let mut out = Vec::with_capacity(length as usize);
            for _ in 0..length {
                out.push(Tensor::randn(
                    [virtual_dim, physical_dim, virtual_dim],
                    (kind, device),
                ));
            }
            out
        }
    }
}

pub fn calc_global_tensor_by_contract(mps: &[Tensor]) -> Tensor {
    // Build a single named-einsum that joins all 3-d tensors along virtual bonds.
    let n = mps.len();
    let mut inputs: Vec<String> = Vec::with_capacity(n);
    let mut outputs: Vec<String> = Vec::new();
    let mut specs: Vec<Vec<String>> = Vec::new();
    for i in 0..n {
        let labels = vec![format!("t{}0", i), format!("t{}1", i), format!("t{}2", i)];
        inputs.push(labels.join(" "));
        specs.push(labels);
    }
    let mps_type = get_mps_type(mps);
    if mps_type == MPSType::Periodic {
        // contract right of i with left of i+1 and last right with first left
        // Output keeps only all physical dims t{i}1
        outputs = (0..n).map(|i| format!("t{}1", i)).collect();
    } else {
        // Open: keep first left and last right too
        outputs.push("t00".to_string());
        outputs.extend((0..n).map(|i| format!("t{}1", i)));
        outputs.push(format!("t{}2", n - 1));
    }
    let spec = format!("{} -> {}", inputs.join(", "), outputs.join(" "));
    let owned: Vec<Tensor> = mps.iter().map(|t| t.shallow_clone()).collect();
    named_einsum(&spec, &owned).squeeze()
}

pub fn calc_global_tensor_by_tensordot(mps: &[Tensor]) -> Tensor {
    assert!(!mps.is_empty());
    let mut state = mps[0].shallow_clone();
    for next in mps.iter().skip(1) {
        let next = next.shallow_clone();
        state = Tensor::einsum("... r, r p v -> ... p v", &[state, next], None::<Vec<i64>>);
    }
    state.squeeze()
}

pub fn project_multi_qubits_vec(
    mps_local_tensors: &[Tensor],
    qubit_indices: &[i64],
    project_to_states: &Tensor,
) -> Vec<Tensor> {
    let mut local_tensors: Vec<Tensor> = mps_local_tensors
        .iter()
        .map(|t| t.shallow_clone())
        .collect();
    if qubit_indices.is_empty() {
        return local_tensors;
    }
    assert_eq!(project_to_states.size()[0], qubit_indices.len() as i64);
    for (i, &qidx) in qubit_indices.iter().enumerate() {
        let lt = &local_tensors[qidx as usize];
        let ps = project_to_states.i(i as i64);
        assert_eq!(lt.size()[1], ps.size()[0]);
        let new_lt = Tensor::einsum(
            "l p r, p -> l r",
            &[lt.shallow_clone(), ps],
            None::<Vec<i64>>,
        );
        local_tensors[qidx as usize] = new_lt;
    }
    let mut idxs: Vec<i64> = qubit_indices.to_vec();
    idxs.sort_by(|a, b| b.cmp(a));
    for &qidx in idxs.iter().take(idxs.len().saturating_sub(1)) {
        assert!(qidx > 0);
        let left = (qidx - 1) as usize;
        let right = qidx as usize;
        let merged = Tensor::einsum(
            "a b, b c -> a c",
            &[
                local_tensors[left].shallow_clone(),
                local_tensors[right].shallow_clone(),
            ],
            None::<Vec<i64>>,
        );
        local_tensors[left] = merged;
        let _ = local_tensors.remove(right);
    }
    if local_tensors.len() > 1 {
        let qidx = *idxs.last().unwrap() as usize;
        if qidx == 0 {
            let merged = Tensor::einsum(
                "a b, b c -> a c",
                &[
                    local_tensors[qidx].shallow_clone(),
                    local_tensors[1].shallow_clone(),
                ],
                None::<Vec<i64>>,
            );
            local_tensors[1] = merged;
        } else {
            let left = qidx - 1;
            let merged = Tensor::einsum(
                "a b, b c -> a c",
                &[
                    local_tensors[left].shallow_clone(),
                    local_tensors[qidx].shallow_clone(),
                ],
                None::<Vec<i64>>,
            );
            local_tensors[left] = merged;
        }
        let _ = local_tensors.remove(qidx);
    }
    for lt in &mut local_tensors {
        if lt.dim() == 2 {
            *lt = lt.unsqueeze(1);
        }
    }
    local_tensors
}

pub fn orthogonalize_left2right_step(
    mps_tensors: &[Tensor],
    local_tensor_idx: usize,
    mode: &str,
    truncate_dim: Option<i64>,
    normalize: bool,
    check_nan: bool,
) -> (Tensor, Tensor) {
    assert!(mps_tensors.len() > 1);
    assert!(local_tensor_idx < mps_tensors.len() - 1);
    let mode = mode.to_lowercase();
    assert!(mode == "svd" || mode == "qr");
    let local = &mps_tensors[local_tensor_idx];
    let shape = local.size();
    let right_dim = shape[2];
    let need_truncate = truncate_dim.is_some();
    if need_truncate {
        assert!(mode == "svd");
    }
    let view = local.view([-1, right_dim]);
    let (new_local, r) = if mode == "svd" {
        let (u, s, v) = view.svd(false, true);
        if let Some(td) = truncate_dim {
            let td = td.min(right_dim);
            let u = u.i((.., 0..td));
            let s = s.i(0..td).unsqueeze(1);
            let v = v.i(0..td);
            (u, s * v)
        } else {
            (u, s.unsqueeze(1) * v)
        }
    } else {
        let (u, r) = view.qr(false);
        (u, r)
    };
    let r = if normalize { &r / r.norm() } else { r };
    let new_local_tensor = new_local.view([shape[0], shape[1], -1]);
    let right = &mps_tensors[local_tensor_idx + 1];
    let new_right = Tensor::einsum("ab,bcd->acd", &[r, right.shallow_clone()], None::<Vec<i64>>);
    if check_nan {
        assert!(new_local_tensor.isnan().any().int64_value(&[]) == 0);
        assert!(new_right.isnan().any().int64_value(&[]) == 0);
    }
    (new_local_tensor, new_right)
}

pub fn orthogonalize_right2left_step(
    mps_tensors: &[Tensor],
    local_tensor_idx: usize,
    mode: &str,
    truncate_dim: Option<i64>,
    normalize: bool,
    check_nan: bool,
) -> (Tensor, Tensor) {
    assert!(mps_tensors.len() > 1);
    assert!(local_tensor_idx > 0 && local_tensor_idx < mps_tensors.len());
    let mode = mode.to_lowercase();
    assert!(mode == "svd" || mode == "qr");
    let local = &mps_tensors[local_tensor_idx];
    let shape = local.size();
    let left_dim = shape[0];
    let view = local.view([left_dim, -1]).transpose(0, 1);
    let need_truncate = truncate_dim.is_some();
    if need_truncate {
        assert!(mode == "svd");
    }
    let (q_t, r) = if mode == "svd" {
        let (u, s, v) = view.svd(false, true);
        let (u, s, v, _rank) = if let Some(td) = truncate_dim {
            let rank = s.size()[0].min(td);
            (u.i((.., 0..rank)), s.i(0..rank), v.i(0..rank), rank)
        } else {
            let r = s.size()[0];
            (u, s, v, r)
        };
        (u.transpose(0, 1), s.unsqueeze(1) * v)
    } else {
        let (q, r) = view.qr(false);
        (q.transpose(0, 1), r)
    };
    let r = if normalize { &r / r.norm() } else { r };
    let new_local = q_t.view([-1, shape[1], shape[2]]);
    let left = &mps_tensors[local_tensor_idx - 1];
    let new_left = Tensor::einsum("abc,dc->abd", &[left.shallow_clone(), r], None::<Vec<i64>>);
    if check_nan {
        assert!(new_local.isnan().any().int64_value(&[]) == 0);
        assert!(new_left.isnan().any().int64_value(&[]) == 0);
    }
    (new_left, new_local)
}

#[allow(clippy::too_many_arguments)]
pub fn orthogonalize_arange(
    mps_tensors: &[Tensor],
    start_idx: usize,
    end_idx: usize,
    mode: &str,
    truncate_dim: Option<i64>,
    normalize: bool,
    return_changed: bool,
    check_nan: bool,
) -> (Vec<Tensor>, Option<Vec<usize>>) {
    let n = mps_tensors.len();
    assert!(n > 1);
    assert!(start_idx < n && end_idx < n);
    let mut mps: Vec<Tensor> = mps_tensors.iter().map(|t| t.shallow_clone()).collect();
    let mut changed = std::collections::BTreeSet::new();
    if start_idx < end_idx {
        for idx in start_idx..end_idx {
            let (l, r) =
                orthogonalize_left2right_step(&mps, idx, mode, truncate_dim, normalize, check_nan);
            mps[idx] = l;
            mps[idx + 1] = r;
            changed.insert(idx);
            changed.insert(idx + 1);
        }
    } else if start_idx > end_idx {
        for idx in (end_idx + 1..=start_idx).rev() {
            let (l, r) =
                orthogonalize_right2left_step(&mps, idx, mode, truncate_dim, normalize, check_nan);
            mps[idx - 1] = l;
            mps[idx] = r;
            changed.insert(idx - 1);
            changed.insert(idx);
        }
    }
    if return_changed {
        (mps, Some(changed.into_iter().collect()))
    } else {
        (mps, None)
    }
}

pub fn tt_decomposition(
    state: &Tensor,
    max_rank: Option<i64>,
    use_svd: bool,
) -> (Vec<Tensor>, Vec<i64>) {
    let clip = max_rank.is_some();
    let use_svd = if clip { true } else { use_svd };
    let shape = state.size();
    let n_qubits = state.dim() as i64;
    let physical_dim = shape[0];
    let mut left_dim = 1_i64;
    let mut locals: Vec<Tensor> = Vec::new();
    let mut remained = state.shallow_clone();
    let mut clipped: Vec<i64> = Vec::new();
    for _ in 0..(n_qubits - 1) {
        let mid_dim = physical_dim;
        if use_svd {
            let m = remained.view([left_dim * mid_dim, -1]);
            let (q, s, v) = m.svd(false, true);
            let (q, s, v, rank) = if let Some(maxr) = max_rank {
                let rank = s.size()[0].min(maxr);
                (q.i((.., 0..rank)), s.i(0..rank), v.i(0..rank), rank)
            } else {
                let r = s.size()[0];
                (q, s, v, r)
            };
            let s = s.unsqueeze(1);
            remained = s * v;
            let new_left = rank;
            locals.push(q.view([left_dim, mid_dim, new_left]));
            left_dim = new_left;
            if clip {
                clipped.push(rank);
            }
        } else {
            let m = remained.view([left_dim * mid_dim, -1]);
            let (q, r) = m.qr(false);
            remained = r;
            let new_left = remained.size()[0];
            locals.push(q.view([left_dim, mid_dim, new_left]));
            left_dim = new_left;
        }
    }
    locals.push(remained.view([left_dim, physical_dim, 1]));
    (locals, clipped)
}

pub fn calculate_mps_norm_factors(mps: &[Tensor], efficient_open: bool) -> Tensor {
    assert!(!mps.is_empty());
    let conj: Vec<Tensor> = mps.iter().map(|t| t.conj()).collect();
    let n = mps.len();
    let device = conj[0].device();
    let kind = conj[0].kind();
    match get_mps_type(mps) {
        MPSType::Open => {
            let mut v = Tensor::ones([1, 1], (kind, device)); // a b
            let norms = Tensor::empty([n as i64], (kind, device));
            if efficient_open {
                for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                    v = Tensor::einsum("ab,aix->bix", &[v, _c.shallow_clone()], None::<Vec<i64>>);
                    v = Tensor::einsum("bix,biy->xy", &[v, _m.shallow_clone()], None::<Vec<i64>>);
                    let nf = v.norm();
                    v = &v / &nf;
                    norms.i((i as i64,)).copy_(&nf);
                }
            } else {
                for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                    v = Tensor::einsum(
                        "ab,aix,biy->xy",
                        &[v, _c.shallow_clone(), _m.shallow_clone()],
                        None::<Vec<i64>>,
                    );
                    let nf = v.norm();
                    v = &v / &nf;
                    norms.i((i as i64,)).copy_(&nf);
                }
            }
            norms
        }
        MPSType::Periodic => {
            let vdim = mps[0].size()[0];
            let norms = Tensor::empty([n as i64], (kind, device));
            let mut v = Tensor::eye(vdim * vdim, (kind, device)).view([vdim, vdim, vdim, vdim]);
            for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                v = Tensor::einsum(
                    "uvap,adb,pdq->uvbq",
                    &[v, _c.shallow_clone(), _m.shallow_clone()],
                    None::<Vec<i64>>,
                );
                let nf = v.norm();
                v = &v / &nf;
                norms.i((i as i64,)).copy_(&nf);
            }
            let final_nf = Tensor::einsum("acac->", &[v], None::<Vec<i64>>);
            let last = norms.i((n as i64 - 1,)) * final_nf;
            norms.i((n as i64 - 1,)).copy_(&last);
            norms
        }
    }
}

pub fn normalize_mps(mps: &[Tensor]) -> Vec<Tensor> {
    let norms = calculate_mps_norm_factors(mps, true);
    let factors = 1.0f64 / norms.sqrt();
    let n = mps.len();
    let mut out: Vec<Tensor> = Vec::with_capacity(n);
    for (i, t) in mps.iter().enumerate() {
        let f = factors.double_value(&[i as i64]);
        out.push(t * f);
    }
    out
}

pub fn calc_inner_product(mps0: &[Tensor], mps1: &[Tensor]) -> Tensor {
    assert_eq!(mps0.len(), mps1.len());
    let n = mps0.len();
    let kind = mps0[0].kind();
    let device = mps0[0].device();
    let v0 = Tensor::eye(mps0[0].size()[0], (kind, device));
    let v1 = Tensor::eye(mps1[0].size()[0], (kind, device));
    let mut v = Tensor::einsum("ab,xy->axby", &[v0, v1], None::<Vec<i64>>);
    let factors = Tensor::empty([(n as i64) + 1], (kind, device));
    for (i, (m0, m1)) in mps0.iter().zip(mps1.iter()).enumerate() {
        v = Tensor::einsum(
            "uvap,adb,pdq->uvbq",
            &[v, m0.conj(), m1.shallow_clone()],
            None::<Vec<i64>>,
        );
        let nf = v.norm();
        v = &v / &nf;
        factors.i((i as i64,)).copy_(&nf);
    }
    let last = if v.numel() > 1 {
        Tensor::einsum("acac->", &[v], None::<Vec<i64>>)
    } else {
        v.reshape([1]).i(0)
    };
    factors.i((n as i64,)).copy_(&last);
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_tensor_shape_open() {
        let mps = gen_random_mps_tensors(3, 2, 3, MPSType::Open, Kind::Float, Device::Cpu);
        let g = calc_global_tensor_by_contract(&mps);
        assert_eq!(g.size(), vec![2, 2, 2]);
    }
}
