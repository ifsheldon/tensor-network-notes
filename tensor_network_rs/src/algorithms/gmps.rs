use crate::mps::modules::MPS;
use tch::{IndexOp, Tensor};

const EPS: f64 = 1e-14;

pub fn calc_left_to_right_step(
    current_tensor: &Tensor, // [left, physical, right]
    env_left: &Tensor,       // [batch, left]
    sample: &Tensor,         // [batch, physical]
) -> (Tensor, Tensor) {
    // einsum: batch left, batch physical, left physical right -> batch right
    let next = Tensor::einsum(
        "b l, b p, l p r -> b r",
        &[
            env_left.shallow_clone(),
            sample.shallow_clone(),
            current_tensor.shallow_clone(),
        ],
        None::<Vec<i64>>,
    );
    let norm = next.norm_scalaropt_dim(2.0, [-1].as_slice(), true);
    let denom = &norm + EPS;
    let next_normed = &next / denom;
    (next_normed, norm.squeeze())
}

pub fn calc_right_to_left_step(
    current_tensor: &Tensor, // [left, physical, right]
    env_right: &Tensor,      // [batch, right]
    sample: &Tensor,         // [batch, physical]
) -> (Tensor, Tensor) {
    // einsum: batch right, batch physical, left physical right -> batch left
    let next = Tensor::einsum(
        "b r, b p, l p r -> b l",
        &[
            env_right.shallow_clone(),
            sample.shallow_clone(),
            current_tensor.shallow_clone(),
        ],
        None::<Vec<i64>>,
    );
    let norm = next.norm_scalaropt_dim(2.0, [-1].as_slice(), true);
    let denom = &norm + EPS;
    let next_normed = &next / denom;
    (next_normed, norm.squeeze())
}

pub fn calc_nll(norm_factors: &Tensor) -> Tensor {
    // [batch, feature_num]
    -2.0 * (norm_factors.abs() + EPS).log().sum_dim_intlist(
        [1].as_slice(),
        false,
        norm_factors.kind(),
    )
}

pub fn eval_nll(samples: &Tensor, mps: &MPS, return_avg: bool) -> Tensor {
    // samples: [dataset, feature_num, feature_dim]
    assert!(
        samples.dim() == 3,
        "samples must be [dataset, feature, dim]"
    );
    let dataset = samples.size()[0];
    let feature_num = samples.size()[1];
    assert_eq!(feature_num as usize, mps.len());
    let center = mps.center().expect("MPS must have a center");
    let locals = mps.local_tensors();
    let k = samples.kind();
    let dev = samples.device();

    // Initialize env vectors
    let left_dim = locals[0].size()[0];
    let right_dim = locals.last().unwrap().size()[2];
    let mut env_left = Tensor::ones([dataset, left_dim], (k, dev));
    let mut env_right = Tensor::ones([dataset, right_dim], (k, dev));
    // collect norm factors per site
    let mut norms: Vec<Tensor> = (0..feature_num)
        .map(|_| Tensor::zeros([dataset], (k, dev)))
        .collect();

    // convenience to extract sample at index
    let samples_at = |idx: usize| samples.i((.., idx as i64, ..));

    // Left to center-1
    for idx in 0..center {
        let (next, current_norm) =
            calc_left_to_right_step(&locals[idx], &env_left, &samples_at(idx));
        norms[idx] = current_norm;
        env_left = next;
    }
    // Right to center+1
    let mut idx = feature_num as isize - 1;
    while (idx as usize) > center {
        let i = idx as usize;
        let (next, current_norm) = calc_right_to_left_step(&locals[i], &env_right, &samples_at(i));
        norms[i] = current_norm;
        env_right = next;
        idx -= 1;
    }
    // Center norm factor
    let center_tensor = &locals[center];
    let nf_center = Tensor::einsum(
        "l p r, b l, b p, b r -> b",
        &[
            center_tensor.shallow_clone(),
            env_left.shallow_clone(),
            samples_at(center),
            env_right.shallow_clone(),
        ],
        None::<Vec<i64>>,
    );
    norms[center] = nf_center;

    let norms_stacked = Tensor::stack(&norms, 1);
    let nll = calc_nll(&norms_stacked);
    if return_avg {
        nll.mean(nll.kind())
    } else {
        nll
    }
}

// Training loop is sizeable; left as TODO for now.
// pub fn train_gmps(...) -> (Tensor, MPS) { todo!("Implement optimizer loop"); }
