//! Feature maps for tensor-network classifiers and kernels.

use tch::{Kind, Tensor};

/// Apply cossin feature mapping for qubit systems.
pub fn cossin_feature_map(samples: &Tensor, theta: f64, check_range: bool) -> Tensor {
    let samples = if samples.dim() == 1 {
        samples.unsqueeze(0)
    } else {
        samples.shallow_clone()
    };
    if check_range {
        assert!(
            samples
                .ge(0.0)
                .logical_and(&samples.le(1.0))
                .all()
                .int64_value(&[])
                != 0,
            "Samples should be between 0 and 1. This is usually required. To override this, set check_range=false."
        );
    }
    let angle = samples * (theta * std::f64::consts::PI);
    Tensor::stack(&[angle.cos(), angle.sin()], -1)
}

/// Convert a cossin feature tensor to a batch of qubit state tensors.
pub fn feature_map_to_qubit_state(features: &Tensor) -> Tensor {
    let shape = features.size();
    assert!(
        shape.len() == 3 && shape[2] == 2,
        "feature must be a 3D tensor of shape (batch_size, feature_dim, 2), but got {shape:?}"
    );
    let batch = shape[0];
    let mut state = features.select(1, 0);
    for idx in 1..shape[1] {
        let next = features.select(1, idx);
        let state_dims = state.dim() as i64;
        state = state.unsqueeze(state_dims) * next.view([batch, 1, 2]);
        let mut view_shape = state.size();
        view_shape[0] = batch;
        state = state.reshape(view_shape);
    }
    state
}

/// Apply the linear feature mapping `[x, 1 - x]`.
pub fn linear_mapping(samples: &Tensor) -> Tensor {
    let samples = if samples.dim() == 1 {
        samples.reshape([1, -1])
    } else {
        samples.shallow_clone()
    };
    Tensor::stack(
        &[
            samples.shallow_clone(),
            Tensor::ones_like(&samples) - &samples,
        ],
        -1,
    )
}

/// Return the default floating kind used by feature maps when needed.
pub fn default_feature_kind() -> Kind {
    Kind::Float
}
