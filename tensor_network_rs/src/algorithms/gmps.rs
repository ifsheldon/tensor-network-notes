use crate::constants::NO_OPT_PATH;
use crate::{mps::modules::MPS, types::*};
use tch::{IndexOp, Kind, Tensor};

const EPS: f64 = 1e-14;

/// One sweep step from left to right: updates left environment and returns
/// the normalized next env along with its norm factor, matching Python.
pub fn calc_left_to_right_step(
    current_tensor: &Tensor,          // [left, physical, right]
    current_env_vector_left: &Tensor, // [batch, left]
    current_sample: &Tensor,          // [batch, physical]
) -> (Tensor, Tensor) {
    // einsum: batch left, batch physical, left physical right -> batch right
    let next_env_vector_left = Tensor::einsum(
        "b l, b p, l p r -> b r",
        &[
            current_env_vector_left.shallow_clone(),
            current_sample.shallow_clone(),
            current_tensor.shallow_clone(),
        ],
        NO_OPT_PATH,
    );
    let current_norm_factor = next_env_vector_left.norm_scalaropt_dim(2.0, [-1].as_slice(), true);
    let denom = &current_norm_factor + EPS;
    let normalized_env_vector_left = &next_env_vector_left / denom;
    (normalized_env_vector_left, current_norm_factor.squeeze())
}

/// One sweep step from right to left: symmetric to the left-to-right step.
pub fn calc_right_to_left_step(
    current_tensor: &Tensor,           // [left, physical, right]
    current_env_vector_right: &Tensor, // [batch, right]
    current_sample: &Tensor,           // [batch, physical]
) -> (Tensor, Tensor) {
    // einsum: batch right, batch physical, left physical right -> batch left
    let next_env_vector_right = Tensor::einsum(
        "b r, b p, l p r -> b l",
        &[
            current_env_vector_right.shallow_clone(),
            current_sample.shallow_clone(),
            current_tensor.shallow_clone(),
        ],
        NO_OPT_PATH,
    );
    let current_norm_factor = next_env_vector_right.norm_scalaropt_dim(2.0, [-1].as_slice(), true);
    let denom = &current_norm_factor + EPS;
    let normalized_env_vector_right = &next_env_vector_right / denom;
    (normalized_env_vector_right, current_norm_factor.squeeze())
}

/// Compute negative log-likelihood from per-site norm factors `[batch, L]`.
pub fn calc_nll(norm_factors: &Tensor) -> Tensor {
    // [batch, feature_num]
    -2.0 * (norm_factors.abs() + EPS).log().sum_dim_intlist(
        [1].as_slice(),
        false,
        norm_factors.kind(),
    )
}

/// Evaluate the negative log-likelihood for an MPS given feature-mapped samples.
/// When `return_avg` is true, returns the average NLL over the batch, as in Python.
pub fn eval_nll(samples: &Tensor, mps: &MPS, return_avg: bool) -> Tensor {
    // samples: [dataset, feature_num, feature_dim]
    assert!(
        samples.dim() == 3,
        "samples must be [dataset, feature, dim]"
    );
    let dataset_size = samples.size()[0];
    let feature_num: Num = samples.size()[1].cast();
    assert_eq!(feature_num, mps.len());
    let center = mps.center().expect("MPS must have a center");
    let locals = mps.local_tensors();
    let k = samples.kind();
    let dev = samples.device();

    // Initialize env vectors
    let left_virtual_dim = locals[0].size()[0];
    let right_virtual_dim = locals.last().unwrap().size()[2];
    let mut env_vector_left = Tensor::ones([dataset_size, left_virtual_dim], (k, dev));
    let mut env_vector_right = Tensor::ones([dataset_size, right_virtual_dim], (k, dev));
    // collect norm factors per site
    let mut norm_factors: Vec<Tensor> = (0..feature_num)
        .map(|_| Tensor::zeros([dataset_size], (k, dev)))
        .collect();

    // convenience to extract sample at index
    let samples_at = |idx: usize| samples.i((.., idx as TInt, ..));

    // Left to center-1
    for idx in 0..center {
        let idx: usize = idx.cast();
        let (next_env_vector_left, current_norm_factor) =
            calc_left_to_right_step(&locals[idx], &env_vector_left, &samples_at(idx));
        norm_factors[idx] = current_norm_factor;
        env_vector_left = next_env_vector_left;
    }
    // Right to center+1
    let mut idx = feature_num - 1;
    while idx > center {
        let i = idx as usize;
        let (next_env_vector_right, current_norm_factor) =
            calc_right_to_left_step(&locals[i], &env_vector_right, &samples_at(i));
        norm_factors[i] = current_norm_factor;
        env_vector_right = next_env_vector_right;
        idx -= 1;
    }
    // Center norm factor
    let center_index: usize = center.cast();
    let center_tensor = &locals[center_index];
    let center_norm_factor = Tensor::einsum(
        "l p r, b l, b p, b r -> b",
        &[
            center_tensor.shallow_clone(),
            env_vector_left.shallow_clone(),
            samples_at(center_index),
            env_vector_right.shallow_clone(),
        ],
        NO_OPT_PATH,
    );
    norm_factors[center_index] = center_norm_factor;

    let norm_factors_stacked = Tensor::stack(&norm_factors, 1);
    let nll = calc_nll(&norm_factors_stacked);
    if return_avg {
        nll.mean(nll.kind())
    } else {
        nll
    }
}

/// Evaluate NLL over a selected subset of feature positions.
/// Note: for now, when `indices` covers all positions, this defers to `eval_nll`.
/// For partial subsets, this currently returns an error (TODO: implement full matrix-env version).
pub fn eval_nll_selected_features(
    samples: &Tensor,
    mps: &MPS,
    indices: &[UIdx],
    return_avg: bool,
) -> Tensor {
    let feature_num: Num = samples.size()[1].cast();
    // validate indices
    let mut set = std::collections::BTreeSet::new();
    for &i in indices {
        assert!(i < feature_num);
        set.insert(i as usize);
    }
    if set.len() == feature_num as usize {
        return eval_nll(samples, mps, return_avg);
    }

    // Shapes and helpers
    let dataset_size = samples.size()[0];
    let k = samples.kind();
    let dev = samples.device();
    let center: usize = mps.center().expect("MPS must have a center").cast();
    let locals = mps.local_tensors();

    // env tensors as [batch, L, R] where L/R dims vary along sweep
    let mut env_vectors_left = Tensor::ones([dataset_size, 1, 1], (k, dev));
    let mut env_vectors_right = Tensor::ones([dataset_size, 1, 1], (k, dev));
    let norm_factors = Tensor::ones([dataset_size, feature_num.cast()], (k, dev));

    let in_subset = |i: usize| set.contains(&i);
    let sample_at = |idx: usize| samples.i((.., idx.to_tint(), ..));

    // left to center-1
    for (i, _t) in locals.iter().enumerate().take(center) {
        let local_tensor = &locals[i]; // [l,p,r]
        env_vectors_left = if in_subset(i) {
            // selected: contract with sample -> [b,l,r]
            Tensor::einsum(
                "l p r, b p -> b l r",
                &[local_tensor.shallow_clone(), sample_at(i)],
                NO_OPT_PATH,
            )
        } else {
            // unselected: transfer env: E = A^† E A -> [b, r*, r]
            Tensor::einsum(
                "lC p rC, b lC l, l p r -> b rC r",
                &[
                    local_tensor.conj(),
                    env_vectors_left.shallow_clone(),
                    local_tensor.shallow_clone(),
                ],
                NO_OPT_PATH,
            )
        };
        let current_norm_factor = env_vectors_left
            .copy()
            .norm_scalaropt_dim(2.0, [-1, -2].as_slice(), false)
            .squeeze(); // [b]
        norm_factors
            .i((.., i.to_tint()))
            .copy_(&current_norm_factor);
        env_vectors_left = &env_vectors_left / (current_norm_factor.view([-1, 1, 1]) + EPS);
    }

    // right to center+1
    let mut i = feature_num as isize - 1;
    while (i as usize) > center {
        let ui = i as usize;
        let local_tensor = &locals[ui];
        env_vectors_right = if in_subset(ui) {
            // map to [b, l, r] then contract as right-env build requires later
            Tensor::einsum(
                "l p r, b p -> b l r",
                &[local_tensor.shallow_clone(), sample_at(ui)],
                NO_OPT_PATH,
            )
        } else {
            Tensor::einsum(
                "lC p rC, b rC r, l p r -> b lC l",
                &[
                    local_tensor.conj(),
                    env_vectors_right.shallow_clone(),
                    local_tensor.shallow_clone(),
                ],
                NO_OPT_PATH,
            )
        };
        let current_norm_factor = env_vectors_right
            .copy()
            .norm_scalaropt_dim(2.0, [-1, -2].as_slice(), false)
            .squeeze();
        norm_factors
            .i((.., ui.to_tint()))
            .copy_(&current_norm_factor);
        env_vectors_right = &env_vectors_right / (current_norm_factor.view([-1, 1, 1]) + EPS);
        i -= 1;
    }

    // center contribution
    let center_tensor = &locals[center]; // [l,p,r]
    let center_norm_factor = if in_subset(center) {
        // contract sample first -> [b,l,r]
        let new_center_tensor = Tensor::einsum(
            "l p r, b p -> b l r",
            &[center_tensor.shallow_clone(), sample_at(center)],
            NO_OPT_PATH,
        );
        Tensor::einsum(
            "b lC l, b lC rC, b l r, b rC r -> b",
            &[
                env_vectors_left.shallow_clone(),
                new_center_tensor.conj(),
                new_center_tensor,
                env_vectors_right.shallow_clone(),
            ],
            NO_OPT_PATH,
        )
        .abs()
    } else {
        Tensor::einsum(
            "lC p rC, l p r, b lC l, b rC r -> b",
            &[
                center_tensor.conj(),
                center_tensor.shallow_clone(),
                env_vectors_left,
                env_vectors_right,
            ],
            NO_OPT_PATH,
        )
        .abs()
    };
    norm_factors
        .i((.., center.to_tint()))
        .copy_(&center_norm_factor);

    let nll = calc_nll(&norm_factors);
    if return_avg { nll.mean(k) } else { nll }
}

/// Calculate the gradient with respect to the current local tensor.
/// Implements the same normalized gradient expression and optional TSGO
/// projection used in the Python code.
pub fn calc_gradient(
    env_left: &Tensor,
    env_right: &Tensor,
    sample: &Tensor,
    current_tensor: &Tensor,
    enable_tsgo: bool,
) -> Tensor {
    let raw_grad = Tensor::einsum(
        "b l, b p, b r -> b l p r",
        &[
            env_left.shallow_clone(),
            sample.shallow_clone(),
            env_right.shallow_clone(),
        ],
        NO_OPT_PATH,
    );
    let norm = Tensor::einsum(
        "l p r, b l p r -> b",
        &[current_tensor.shallow_clone(), raw_grad.shallow_clone()],
        NO_OPT_PATH,
    );
    let sign = norm.sign();
    let norm = norm + sign * EPS;
    let grad_part = (raw_grad / norm.view([-1, 1, 1, 1])).mean_dim(
        [0].as_slice(),
        false,
        current_tensor.kind(),
    );
    let mut grad: Tensor = 2.0 * (current_tensor - &grad_part);
    if enable_tsgo {
        let g_flat = grad.view([-1]);
        let w_flat = current_tensor.view([-1]);
        let proj = g_flat.dot(&w_flat) * w_flat;
        let size = current_tensor.size();
        grad = (g_flat - proj).view(size.as_slice());
    }
    let norm = grad.norm();
    &grad / norm
}

/// Train an MPS with the GMPS algorithm. Returns the loss curve and the trained MPS.
/// Follows the two-sweep (L→R then R→L) update pattern in the Python version.
pub fn train_gmps(
    samples: &Tensor,
    batch_size: Num,
    mut mps: MPS,
    sweep_times: Num,
    lr: f64,
    enable_tsgo: bool,
) -> (Tensor, MPS) {
    let dataset_size = samples.size()[0].cast();
    assert!(dataset_size % batch_size == 0);
    mps.center_orthogonalization(0, "qr", None, true, true);
    let init_nll = eval_nll(samples, &mps, true);
    let mut losses: Vec<Tensor> = vec![init_nll];
    let feature_num = samples.size()[1].cast();
    let k = samples.kind();
    let dev = samples.device();
    for _ in 0..sweep_times {
        let mut epoch_losses: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < dataset_size {
            let end = (start + batch_size).min(dataset_size);
            let batch = samples.i(start.to_tint()..end.to_tint());
            let bsz = batch.size()[0];
            // Prepare env vectors
            let left_dim = mps.local_tensors()[0].size()[0];
            let right_dim = mps.local_tensors().last().unwrap().size()[2];
            let mut env_left: Vec<Option<Tensor>> = (0..feature_num).map(|_| None).collect();
            env_left[0] = Some(Tensor::ones([bsz, left_dim], (k, dev)));
            let mut env_right: Vec<Option<Tensor>> = (0..feature_num).map(|_| None).collect();
            env_right[feature_num - 1] = Some(Tensor::ones([bsz, right_dim], (k, dev)));

            let data_at = |idx: usize| batch.i((.., idx.to_tint(), ..));
            // Right-to-left prepare
            let center: usize = mps.center().unwrap().cast();
            let locals_now = mps.local_tensors();
            let mut idx = feature_num - 1;
            while idx > center {
                let i = idx;
                let (next_r, _nf) = calc_right_to_left_step(
                    &locals_now[i],
                    env_right[i].as_ref().unwrap(),
                    &data_at(i),
                );
                env_right[i - 1] = Some(next_r);
                idx -= 1;
            }
            // Left to right
            for i in 0..feature_num {
                let center: usize = mps.center().unwrap().cast();
                assert_eq!(i, center);
                let locals_now = mps.local_tensors();
                let l_env = env_left[i].as_ref().unwrap();
                let fallback_r = Tensor::ones([bsz, locals_now[i].size()[2]], (k, dev));
                let r_env = env_right[i].as_ref().unwrap_or(&fallback_r);
                let grad = calc_gradient(l_env, r_env, &data_at(i), &locals_now[i], enable_tsgo);
                mps.force_set_local_tensor(i, &locals_now[i] - lr * &grad);
                if i < feature_num - 1 {
                    mps.center_orthogonalization((i + 1).cast(), "qr", None, true, true);
                    let locals_now = mps.local_tensors();
                    let (next_l, _nf) = calc_left_to_right_step(
                        &locals_now[i],
                        env_left[i].as_ref().unwrap(),
                        &data_at(i),
                    );
                    env_left[i + 1] = Some(next_l);
                } else {
                    mps.center_normalize();
                }
            }
            // Right to left
            for i in (0..feature_num).rev() {
                let center: usize = mps.center().unwrap().cast();
                assert_eq!(i, center);
                let locals_now = mps.local_tensors();
                let fallback_l = Tensor::ones([bsz, locals_now[i].size()[0]], (k, dev));
                let l_env = env_left[i].as_ref().unwrap_or(&fallback_l);
                let r_env = env_right[i].as_ref().unwrap();
                let grad = calc_gradient(l_env, r_env, &data_at(i), &locals_now[i], enable_tsgo);
                mps.force_set_local_tensor(i, &locals_now[i] - lr * &grad);
                if i > 0 {
                    mps.center_orthogonalization((i - 1).cast(), "qr", None, true, true);
                    let locals_now = mps.local_tensors();
                    let (next_r, _nf) = calc_right_to_left_step(
                        &locals_now[i],
                        env_right[i].as_ref().unwrap(),
                        &data_at(i),
                    );
                    env_right[i - 1] = Some(next_r);
                } else {
                    mps.center_normalize();
                }
            }
            let loss = eval_nll(&batch, &mps, true);
            epoch_losses.push(loss);
            start = end;
        }
        let epoch = Tensor::stack(&epoch_losses, 0).mean(epoch_losses[0].kind());
        losses.push(epoch);
    }
    (Tensor::stack(&losses, 0), mps)
}

/// Generate a sample using a GMPS by sequentially measuring sites.
/// If `sample` and `gen_indices` are provided, projects non-generated sites using the provided feature mapping
/// and generates only on `gen_indices`. Returns the average over `sample_num` draws as a length-L float tensor in [0,1].
pub fn generate_sample_with_gmps(
    mps: &MPS,
    sample: Option<&Tensor>, // [L] or [1,L] with values in [0,1] for feature mapping
    sample_num: Num,
    gen_indices: Option<&[UIdx]>,
    gen_order: &str,       // "ascending" | "descending"
    feature_mapping: &str, // currently only "cossin"
    theta: f64,
) -> Tensor {
    assert!(sample_num > 0);
    let length: Num = mps.len().cast();
    let k = mps.local_tensors()[0].kind();
    let dev = mps.local_tensors()[0].device();
    let mut gen_list: Vec<UIdx> = if let Some(idx) = gen_indices {
        idx.to_vec()
    } else {
        (0..length).collect()
    };
    if gen_order == "descending" {
        gen_list.reverse();
    } else {
        assert!(gen_order == "ascending");
    }
    assert!(
        feature_mapping == "cossin",
        "only cossin mapping is supported for now"
    );

    // Prepare projected MPS if partial info is provided
    let (base_mps, remaining_positions) = if let Some(s) = sample {
        let s = if s.dim() == 2 {
            s.squeeze_dim(0)
        } else {
            s.shallow_clone()
        };
        assert_eq!(s.dim(), 1);
        assert_eq!(s.size()[0] as Num, length);
        // project indices = all \ gen_list
        let mut project_idx: Vec<UIdx> = (0..length).collect();
        for &g in &gen_list {
            project_idx.retain(|&x| x != g);
        }
        if project_idx.is_empty() {
            let cloned: Vec<Tensor> = mps
                .local_tensors()
                .iter()
                .map(|t| t.shallow_clone())
                .collect();
            (MPS::from_tensors(cloned, Some(false)), gen_list.clone())
        } else {
            // feature mapping for projection positions
            let mut vals: Vec<Tensor> = Vec::with_capacity(project_idx.len());
            for &pi in &project_idx {
                vals.push(s.i(pi.to_tint()));
            }
            let proj_vec = Tensor::stack(&vals, 0).to_kind(k).to_device(dev); // [P]
            let feats =
                crate::feature_mapping::cossin_feature_map(&proj_vec.unsqueeze(0), theta, false)
                    .squeeze_dim(0); // [P,2]
            let base = mps.project_multi_qubits_vec(&project_idx, &feats);
            // remaining positions are exactly gen_list in original order, map them to new indices [0..L-P)
            // After projection, the remaining sites preserve relative order; new index = position within the sorted remaining set
            let mut rem: Vec<UIdx> = (0..length).collect();
            for &pi in &project_idx {
                rem.retain(|&x| x != pi);
            }
            let remaining_positions = rem;
            // rem maps new index -> original index
            // gen_list refers to original indices; ensure all are in rem
            for &g in &gen_list {
                assert!(remaining_positions.contains(&g));
            }
            (base, remaining_positions)
        }
    } else {
        let cloned: Vec<Tensor> = mps
            .local_tensors()
            .iter()
            .map(|t| t.shallow_clone())
            .collect();
        (
            MPS::from_tensors(cloned, Some(false)),
            (0..length).collect(),
        )
    };

    // Map original gen indices to new indices in the (possibly) projected MPS
    let gen_new: Vec<UIdx> = gen_list
        .iter()
        .map(|&g| {
            remaining_positions
                .iter()
                .position(|&x| x == g)
                .unwrap()
                .cast()
        })
        .collect();

    // accumulator over runs
    let mut acc = Tensor::zeros([length.to_tint()], (Kind::Float, tch::Device::Cpu));
    for _ in 0..sample_num {
        let cloned: Vec<Tensor> = base_mps
            .local_tensors()
            .iter()
            .map(|t| t.shallow_clone())
            .collect();
        let mut mps_i = MPS::from_tensors(cloned, Some(false));
        let mut rem_i = remaining_positions.clone();
        let mut gen_i = gen_new.clone();
        let out = if let Some(s) = sample {
            s.to_device(tch::Device::Cpu).to_kind(Kind::Float)
        } else {
            Tensor::zeros([length.to_tint()], (Kind::Float, tch::Device::Cpu))
        };
        while !gen_i.is_empty() {
            let gen_idx: usize = gen_i[0].cast(); // index in current MPS
            let gen_orig = rem_i[gen_idx]; // original position
            mps_i.center_orthogonalization(gen_idx.cast(), "qr", None, true, true);
            let rdm = mps_i.one_body_reduced_density_matrix(gen_idx.cast(), true, true);
            let p1 = rdm.double_value(&[1, 1]);
            let pt = Tensor::from(p1).to_kind(Kind::Float);
            let state = pt.bernoulli().to_kind(Kind::Int64).int64_value(&[]).cast();
            // set output at original position
            out.i(gen_orig.to_tint()).copy_(&Tensor::from(state as f64));
            // project site and update indices
            let new_mps = mps_i.project_multi_qubits_indices(&[gen_idx.cast()], &[state]);
            mps_i = new_mps;
            mps_i.center_orthogonalization((gen_idx - 1).cast(), "qr", None, true, true);
            // remove this position from remaining positions and update mapping
            rem_i.remove(gen_idx);
            gen_i.remove(0);
            for g in &mut gen_i {
                if (*g as usize) > gen_idx {
                    *g -= 1;
                }
            }
        }
        acc += &out;
    }
    &acc / (sample_num as f64)
}

/// Classify by picking the MPS with minimum NLL over all features.
pub fn gmps_classify(samples: &Tensor, gmpss: &[MPS]) -> Tensor {
    assert!(samples.dim() == 3);
    let mut nlls: Vec<Tensor> = Vec::with_capacity(gmpss.len());
    for g in gmpss {
        nlls.push(eval_nll(samples, g, false));
    }
    let mat = Tensor::stack(&nlls, 1); // [B, G]
    mat.argmin(1, false).to_kind(Kind::Int64)
}

/// Classify using only a subset of features.
pub fn gmps_classify_with_selected_features(
    samples: &Tensor,
    gmpss: &[MPS],
    indices: &[UIdx],
) -> Tensor {
    assert!(samples.dim() == 3);
    let mut nlls: Vec<Tensor> = Vec::with_capacity(gmpss.len());
    for g in gmpss {
        nlls.push(eval_nll_selected_features(samples, g, indices, false));
    }
    let mat = Tensor::stack(&nlls, 1);
    mat.argmin(1, false).to_kind(Kind::Int64)
}
