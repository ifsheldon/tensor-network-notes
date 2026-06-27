//! Generative MPS algorithms.

use tch::{Device, IndexOp, Kind, Tensor};

use crate::feature_mapping::cossin_feature_map;
use crate::mps::{MPS, ProjectToStates};
use crate::types::{MPSType, OrthogonalizationMode};

const EPS: f64 = 1e-14;

/// Calculate one left-to-right environment step.
pub fn calc_left_to_right_step(
    current_tensor: &Tensor,
    current_env_vector_left: &Tensor,
    current_sample: &Tensor,
) -> (Tensor, Tensor) {
    let next = Tensor::einsum(
        "bl,bp,lpr->br",
        &[current_env_vector_left, current_sample, current_tensor],
        None::<i64>,
    );
    let norm = next.norm_scalaropt_dim(2.0, [1].as_slice(), true);
    (&next / (&norm + EPS), norm.squeeze_dim(-1))
}

/// Calculate one right-to-left environment step.
pub fn calc_right_to_left_step(
    current_tensor: &Tensor,
    current_env_vector_right: &Tensor,
    current_sample: &Tensor,
) -> (Tensor, Tensor) {
    let next = Tensor::einsum(
        "br,bp,lpr->bl",
        &[current_env_vector_right, current_sample, current_tensor],
        None::<i64>,
    );
    let norm = next.norm_scalaropt_dim(2.0, [1].as_slice(), true);
    (&next / (&norm + EPS), norm.squeeze_dim(-1))
}

/// Calculate negative log likelihood from norm factors.
pub fn calc_nll(norm_factors: &Tensor) -> Tensor {
    -2.0 * (norm_factors.abs() + EPS)
        .log()
        .sum_dim_intlist([1].as_slice(), false, None::<Kind>)
}

/// Calculate the GMPS gradient for the current tensor.
pub fn calc_gradient(
    env_left_vector: &Tensor,
    env_right_vector: &Tensor,
    current_sample: &Tensor,
    current_tensor: &Tensor,
    enable_tsgo: bool,
) -> Tensor {
    let raw_grad = Tensor::einsum(
        "bl,bp,br->blpr",
        &[env_left_vector, current_sample, env_right_vector],
        None::<i64>,
    );
    let norm = Tensor::einsum("lpr,blpr->b", &[current_tensor, &raw_grad], None::<i64>);
    let norm = &norm + norm.sign() * EPS;
    let grad_part =
        (raw_grad / norm.view([-1, 1, 1, 1])).mean_dim([0].as_slice(), false, None::<Kind>);
    let mut grad = (current_tensor - grad_part) * 2.0;
    assert_eq!(grad.size(), current_tensor.size());
    if enable_tsgo {
        let grad_shape = grad.size();
        let grad_flat = grad.reshape([-1]);
        let current_flat = current_tensor.reshape([-1]);
        let projection = grad_flat.dot(&current_flat) * current_flat;
        grad = (grad_flat - projection).reshape(&grad_shape);
    }
    &grad / grad.norm()
}

/// Evaluate the negative log likelihood of feature-mapped samples.
pub fn eval_nll(samples: &Tensor, mps: &MPS, device: Device, return_avg: bool) -> Tensor {
    let samples = samples.to_device(device);
    assert_eq!(
        samples.dim(),
        3,
        "samples must have shape (dataset, feature, dim)"
    );
    let center = mps.center().expect("mps.center must not be None");
    let (dataset_size, feature_num, _) = samples.size3().expect("3D samples");
    assert_eq!(feature_num as usize, mps.len());
    let tensors: Vec<Tensor> = mps
        .local_tensors()
        .into_iter()
        .map(|tensor| tensor.to_device(device))
        .collect();
    let mut env_left = Tensor::ones(
        [dataset_size, tensors[0].size()[0]],
        (samples.kind(), device),
    );
    let mut env_right = Tensor::ones(
        [dataset_size, tensors[mps.len() - 1].size()[2]],
        (samples.kind(), device),
    );
    let mut factors: Vec<Option<Tensor>> = (0..mps.len()).map(|_| None).collect();
    for idx in 0..center {
        let (next, factor) =
            calc_left_to_right_step(&tensors[idx], &env_left, &samples.i((.., idx as i64, ..)));
        factors[idx] = Some(factor);
        env_left = next;
    }
    for idx in (center + 1..mps.len()).rev() {
        let (next, factor) =
            calc_right_to_left_step(&tensors[idx], &env_right, &samples.i((.., idx as i64, ..)));
        factors[idx] = Some(factor);
        env_right = next;
    }
    factors[center] = Some(Tensor::einsum(
        "lpr,bl,bp,br->b",
        &[
            &tensors[center],
            &env_left,
            &samples.i((.., center as i64, ..)),
            &env_right,
        ],
        None::<i64>,
    ));
    let stacked = Tensor::stack(
        &factors
            .into_iter()
            .map(|factor| factor.expect("all factors filled"))
            .collect::<Vec<_>>(),
        1,
    );
    let nll = calc_nll(&stacked);
    if return_avg {
        nll.mean(None::<Kind>)
    } else {
        nll
    }
}

fn check_selected_feature_indices(indices: &[usize], feature_num: usize) {
    let mut seen = vec![false; feature_num];
    for &idx in indices {
        assert!(idx < feature_num, "selected feature index out of range");
        assert!(!seen[idx], "indices must be unique");
        seen[idx] = true;
    }
}

/// Evaluate NLL using selected feature positions while tracing out the rest.
pub fn eval_nll_selected_features(
    samples: &Tensor,
    mps: &MPS,
    indices: &[usize],
    device: Device,
    return_avg: bool,
) -> Tensor {
    let samples = samples.to_device(device);
    assert_eq!(
        samples.dim(),
        3,
        "samples must have shape (dataset, feature, dim)"
    );
    let center = mps.center().expect("mps.center must not be None");
    let (dataset_size, feature_num, _) = samples.size3().expect("3D samples");
    assert_eq!(feature_num as usize, mps.len());
    check_selected_feature_indices(indices, feature_num as usize);
    let tensors = mps
        .local_tensors()
        .into_iter()
        .map(|tensor| tensor.to_device(device))
        .collect::<Vec<_>>();
    let mut env_left = Tensor::ones([dataset_size, 1, 1], (samples.kind(), device));
    let mut env_right = Tensor::ones([dataset_size, 1, 1], (samples.kind(), device));
    let mut factors: Vec<Option<Tensor>> = (0..mps.len()).map(|_| None).collect();
    for idx in 0..center {
        let local = &tensors[idx];
        env_left = if indices.contains(&idx) {
            let projected = Tensor::einsum(
                "lpr,bp->blr",
                &[local, &samples.i((.., idx as i64, ..))],
                None::<i64>,
            );
            Tensor::einsum(
                "blr,blm,bms->brs",
                &[&projected.conj(), &env_left, &projected],
                None::<i64>,
            )
        } else {
            Tensor::einsum(
                "lpr,blm,mps->brs",
                &[&local.conj(), &env_left, local],
                None::<i64>,
            )
        };
        let norm = env_left.norm_scalaropt_dim(2.0, [1, 2].as_slice(), false);
        env_left = &env_left / (norm.reshape([dataset_size, 1, 1]) + EPS);
        factors[idx] = Some(norm);
    }
    for idx in (center + 1..mps.len()).rev() {
        let local = &tensors[idx];
        env_right = if indices.contains(&idx) {
            let projected = Tensor::einsum(
                "lpr,bp->blr",
                &[local, &samples.i((.., idx as i64, ..))],
                None::<i64>,
            );
            Tensor::einsum(
                "blr,brs,bms->blm",
                &[&projected.conj(), &env_right, &projected],
                None::<i64>,
            )
        } else {
            Tensor::einsum(
                "lpr,brs,mps->blm",
                &[&local.conj(), &env_right, local],
                None::<i64>,
            )
        };
        let norm = env_right.norm_scalaropt_dim(2.0, [1, 2].as_slice(), false);
        env_right = &env_right / (norm.reshape([dataset_size, 1, 1]) + EPS);
        factors[idx] = Some(norm);
    }
    let center_tensor = &tensors[center];
    let center_norm = if indices.contains(&center) {
        let projected = Tensor::einsum(
            "lpr,bp->blr",
            &[center_tensor, &samples.i((.., center as i64, ..))],
            None::<i64>,
        );
        Tensor::einsum(
            "bac,bal,blr,bcr->b",
            &[&projected.conj(), &env_left, &projected, &env_right],
            None::<i64>,
        )
        .abs()
    } else {
        Tensor::einsum(
            "apc,lpr,bal,bcr->b",
            &[&center_tensor.conj(), center_tensor, &env_left, &env_right],
            None::<i64>,
        )
        .abs()
    };
    factors[center] = Some(center_norm);
    let stacked = Tensor::stack(
        &factors
            .into_iter()
            .map(|factor| factor.expect("all factors filled"))
            .collect::<Vec<_>>(),
        1,
    );
    let nll = calc_nll(&stacked);
    if return_avg {
        nll.mean(None::<Kind>)
    } else {
        nll
    }
}

/// Train a GMPS model with the sweep algorithm.
pub fn train_gmps(
    samples: &Tensor,
    batch_size: i64,
    mut mps: MPS,
    sweep_times: i64,
    lr: f64,
    device: Device,
    enable_tsgo: bool,
) -> (Tensor, MPS) {
    let samples = samples.to_device(device);
    let dataset_size = samples.size()[0];
    assert_eq!(dataset_size % batch_size, 0);
    assert_eq!(mps.mps_type(), MPSType::Open);
    mps.set_device(device);
    mps.center_orthogonalize(0, OrthogonalizationMode::Qr, None, true, true);
    let mut losses = vec![eval_nll(&samples, &mps, device, true)];
    for _ in 0..sweep_times {
        let permutation = Tensor::randperm(dataset_size, (Kind::Int64, device));
        let mut epoch_losses = Vec::new();
        for batch_start in (0..dataset_size).step_by(batch_size as usize) {
            let indices = permutation.i(batch_start..batch_start + batch_size);
            let batch = samples.index_select(0, &indices);
            sweep_batch(&batch, &mut mps, lr, enable_tsgo);
            epoch_losses.push(eval_nll(&batch, &mps, device, false));
        }
        losses.push(Tensor::cat(&epoch_losses, 0).mean(None::<Kind>));
    }
    (Tensor::stack(&losses, 0), mps)
}

fn sweep_batch(batch: &Tensor, mps: &mut MPS, lr: f64, enable_tsgo: bool) {
    let device = batch.device();
    let current_batch_size = batch.size()[0];
    let mut env_left: Vec<Option<Tensor>> = (0..mps.len()).map(|_| None).collect();
    let mut env_right: Vec<Option<Tensor>> = (0..mps.len()).map(|_| None).collect();
    env_left[0] = Some(Tensor::ones(
        [current_batch_size, mps.local_tensor(0).size()[0]],
        (batch.kind(), device),
    ));
    env_right[mps.len() - 1] = Some(Tensor::ones(
        [
            current_batch_size,
            mps.local_tensor(mps.len() - 1).size()[2],
        ],
        (batch.kind(), device),
    ));
    for idx in (1..mps.len()).rev() {
        let (next, _) = calc_right_to_left_step(
            mps.local_tensor(idx),
            env_right[idx].as_ref().expect("right env"),
            &batch.i((.., idx as i64, ..)),
        );
        env_right[idx - 1] = Some(next);
    }
    for idx in 0..mps.len() {
        assert_eq!(mps.center(), Some(idx));
        let grad = calc_gradient(
            env_left[idx].as_ref().expect("left env"),
            env_right[idx].as_ref().expect("right env"),
            &batch.i((.., idx as i64, ..)),
            mps.local_tensor(idx),
            enable_tsgo,
        );
        mps.replace_local_tensor(idx, mps.local_tensor(idx) - lr * grad);
        if idx + 1 < mps.len() {
            mps.center_orthogonalize(
                (idx + 1) as isize,
                OrthogonalizationMode::Qr,
                None,
                true,
                true,
            );
            let (next, _) = calc_left_to_right_step(
                mps.local_tensor(idx),
                env_left[idx].as_ref().expect("left env"),
                &batch.i((.., idx as i64, ..)),
            );
            env_left[idx + 1] = Some(next);
        } else {
            mps.center_normalize();
        }
    }
    for idx in (0..mps.len()).rev() {
        assert_eq!(mps.center(), Some(idx));
        let grad = calc_gradient(
            env_left[idx].as_ref().expect("left env"),
            env_right[idx].as_ref().expect("right env"),
            &batch.i((.., idx as i64, ..)),
            mps.local_tensor(idx),
            enable_tsgo,
        );
        mps.replace_local_tensor(idx, mps.local_tensor(idx) - lr * grad);
        if idx > 0 {
            mps.center_orthogonalize(
                (idx - 1) as isize,
                OrthogonalizationMode::Qr,
                None,
                true,
                true,
            );
            let (next, _) = calc_right_to_left_step(
                mps.local_tensor(idx),
                env_right[idx].as_ref().expect("right env"),
                &batch.i((.., idx as i64, ..)),
            );
            env_right[idx - 1] = Some(next);
        } else {
            mps.center_normalize();
        }
    }
}

/// Convert labels to fixed-width binary rows.
pub fn labels_to_binary(labels: &Tensor, num_bits: i64) -> Tensor {
    assert_eq!(labels.dim(), 1, "labels must be a 1D tensor");
    assert_eq!(labels.kind(), Kind::Int64, "labels must be a long tensor");
    assert!(
        labels.min().int64_value(&[]) >= 0,
        "labels must be non-negative"
    );
    assert!(
        labels.max().int64_value(&[]) < 2_i64.pow(num_bits as u32),
        "labels must be less than 2 ** num_bits"
    );
    let powers = Tensor::arange_start_step(num_bits - 1, -1, -1, (Kind::Int64, labels.device()));
    labels
        .unsqueeze(1)
        .bitwise_right_shift(&powers)
        .bitwise_and(1)
        .to_kind(Kind::Float)
}

/// Prepend four binary label features to flattened MNIST images.
pub fn prepend_labels(raw_images: &Tensor, labels: &Tensor) -> Tensor {
    assert_eq!(raw_images.dim(), 2);
    assert_eq!(raw_images.size()[0], labels.size()[0]);
    assert_eq!(raw_images.size()[1], 28 * 28);
    let labels = labels.to_device(raw_images.device());
    let bin_labels = labels_to_binary(&labels, 4).to_kind(raw_images.kind());
    Tensor::cat(&[bin_labels, raw_images.shallow_clone()], 1)
}

/// Generate a sample by sequentially measuring a GMPS.
pub fn generate_sample_with_gmps(
    mps: &MPS,
    sample: Option<&Tensor>,
    sample_num: i64,
    gen_indices: Option<&[usize]>,
    theta: f64,
    ascending: bool,
) -> Tensor {
    assert!(sample_num > 0, "sample_num must be positive");
    let length = mps.len();
    let generate_all = sample.is_none() || gen_indices.is_none();
    let gen_indices: Vec<usize> = if let Some(indices) = gen_indices {
        indices.to_vec()
    } else if ascending {
        (0..length).collect()
    } else {
        (0..length).rev().collect()
    };
    let base_sample = if let Some(sample) = sample {
        assert!(sample.size() == [length as i64] || sample.size() == [1, length as i64]);
        sample.squeeze()
    } else {
        Tensor::zeros([length as i64], (mps.kind(), mps.device()))
    };
    let mut work_mps = mps.shallow_clone();
    if !generate_all {
        let mut project_indices: Vec<usize> = (0..length).collect();
        for idx in &gen_indices {
            project_indices.retain(|item| item != idx);
        }
        let gather = Tensor::from_slice(
            &project_indices
                .iter()
                .map(|&x| x as i64)
                .collect::<Vec<_>>(),
        )
        .to_device(base_sample.device());
        let features =
            cossin_feature_map(&base_sample.index_select(0, &gather), theta, true).squeeze();
        work_mps =
            work_mps.project_multi_qubits(&project_indices, ProjectToStates::Vectors(&features));
    }
    let mut samples = Vec::new();
    for _ in 0..sample_num {
        let sample_i = base_sample.shallow_clone();
        let mut mps_i = work_mps.shallow_clone();
        let mut active_indices = if generate_all {
            gen_indices.clone()
        } else {
            argsort_ranks(&gen_indices)
        };
        let mut pos = 0;
        while !active_indices.is_empty() {
            let gen_idx = active_indices[0];
            mps_i.center_orthogonalize(
                gen_idx as isize,
                OrthogonalizationMode::Qr,
                None,
                true,
                true,
            );
            let rdm = mps_i.one_body_reduced_density_matrix(gen_idx, true, true);
            assert_eq!(rdm.size(), vec![2, 2]);
            let p1 = rdm.diag(0).i(1);
            let p1_value = p1.double_value(&[]);
            assert!(
                (0.0..=1.0).contains(&p1_value),
                "probability should be between 0 and 1"
            );
            let state = p1.bernoulli().to_kind(Kind::Int64).int64_value(&[]);
            let original_idx = gen_indices[pos] as i64;
            let _ = sample_i.i(original_idx).fill_(state as f64);
            let mut new_mps = mps_i.project_one_qubit_to_index(gen_idx, state);
            new_mps.set_center(Some(gen_idx.saturating_sub(1).min(new_mps.len() - 1)));
            mps_i = new_mps;
            for idx in &mut active_indices {
                if *idx > gen_idx {
                    *idx -= 1;
                }
            }
            active_indices.remove(0);
            pos += 1;
        }
        samples.push(sample_i);
    }
    Tensor::stack(&samples, 0).mean_dim([0].as_slice(), false, None::<Kind>)
}

/// Classify data with one GMPS per class.
pub fn gmps_classify(gmpss: &[MPS], data: &Tensor) -> Tensor {
    assert!(!gmpss.is_empty(), "No GMPSs provided");
    assert_eq!(
        data.dim(),
        3,
        "Data must be a 3D tensor of shape (batch, feature_num, feature_dim)"
    );
    assert_eq!(
        data.size()[1] as usize,
        gmpss[0].len(),
        "Feature number mismatch"
    );
    let nlls: Vec<Tensor> = gmpss
        .iter()
        .map(|gmps| eval_nll(data, gmps, gmps.device(), false))
        .collect();
    Tensor::stack(&nlls, 1).argmin(1, false)
}

/// Classify data with selected feature positions.
pub fn gmps_classify_with_selected_features(
    gmpss: &[MPS],
    data: &Tensor,
    indices: &[usize],
) -> Tensor {
    assert!(!gmpss.is_empty(), "No GMPSs provided");
    assert_eq!(
        data.dim(),
        3,
        "Data must be a 3D tensor of shape (batch, feature_num, feature_dim)"
    );
    let feature_num = data.size()[1] as usize;
    assert_eq!(feature_num, gmpss[0].len(), "Feature number mismatch");
    check_selected_feature_indices(indices, feature_num);
    if indices.len() == feature_num {
        return gmps_classify(gmpss, data);
    }
    let nlls = gmpss
        .iter()
        .map(|gmps| eval_nll_selected_features(data, gmps, indices, gmps.device(), false))
        .collect::<Vec<_>>();
    Tensor::stack(&nlls, 1).argmin(1, false)
}

fn argsort_ranks(indices: &[usize]) -> Vec<usize> {
    let mut pairs: Vec<(usize, usize)> = indices.iter().copied().enumerate().collect();
    pairs.sort_by_key(|(_, value)| *value);
    let mut ranks = vec![0; indices.len()];
    for (rank, (original, _)) in pairs.into_iter().enumerate() {
        ranks[original] = rank;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn selected_feature_nll_and_classifier_return_batch_outputs() {
        let mut mps = MPS::random(4, 2, 2, MPSType::Open, Kind::Float, Device::Cpu, false);
        mps.center_orthogonalize(1, OrthogonalizationMode::Qr, None, true, true);
        let raw = Tensor::rand([3, 4], (Kind::Float, Device::Cpu));
        let samples = cossin_feature_map(&raw, 0.5, true);
        let nll = eval_nll_selected_features(&samples, &mps, &[0, 2], Device::Cpu, false);
        assert_eq!(nll.size(), vec![3]);
        assert_eq!(nll.isnan().any().int64_value(&[]), 0);

        let predictions = gmps_classify_with_selected_features(
            &[mps.shallow_clone(), mps.shallow_clone()],
            &samples,
            &[0, 2],
        );
        assert_eq!(predictions.size(), vec![3]);
    }
}
