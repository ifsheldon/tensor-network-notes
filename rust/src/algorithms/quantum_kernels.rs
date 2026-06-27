//! Quantum-kernel distance helpers.

use tch::{Kind, Tensor};

fn check_samples(samples: &Tensor) {
    assert_eq!(samples.dim(), 2, "samples must be a 2D tensor");
    assert!(
        samples.min().double_value(&[]) >= 0.0,
        "samples.min(): {}",
        samples.min().double_value(&[])
    );
    assert!(
        samples.max().double_value(&[]) <= 1.0,
        "samples.max(): {}",
        samples.max().double_value(&[])
    );
}

/// Pairwise negative log cossin metric matrix.
pub fn metric_matrix_neg_log_cos_sin(samples: &Tensor, theta: f64, deduplicate: bool) -> Tensor {
    check_samples(samples);
    assert!(std::f64::consts::PI / 2.0 >= theta && theta >= 0.0);
    let sample_num = samples.size()[0];
    if deduplicate {
        let metric = Tensor::zeros([sample_num, sample_num], (samples.kind(), samples.device()));
        for n in 0..sample_num - 1 {
            let sample_n = samples.get(n).reshape([1, -1]);
            let others = samples.narrow(0, n + 1, sample_num - n - 1);
            let distances = -((sample_n - others) * theta).cos().log().mean_dim(
                [1].as_slice(),
                false,
                None::<Kind>,
            );
            assert!(
                distances.isnan().any().int64_value(&[]) == 0,
                "if there's nan, try to reduce theta"
            );
            let mut upper = metric.narrow(0, n, 1).narrow(1, n + 1, sample_num - n - 1);
            upper.copy_(&distances.reshape([1, -1]));
            let mut lower = metric.narrow(0, n + 1, sample_num - n - 1).narrow(1, n, 1);
            lower.copy_(&distances.reshape([-1, 1]));
        }
        metric
    } else {
        let lhs = samples.unsqueeze(0);
        let rhs = lhs.transpose(0, 1);
        let metric =
            -((lhs - rhs) * theta)
                .cos()
                .log()
                .mean_dim([2].as_slice(), false, None::<Kind>);
        assert!(
            metric.isnan().any().int64_value(&[]) == 0,
            "if there's nan, try to reduce theta"
        );
        metric
    }
}

/// Negative log cossin metric between samples and references.
pub fn metric_neg_log_cos_sin(samples: &Tensor, reference_samples: &Tensor, theta: f64) -> Tensor {
    check_samples(samples);
    check_samples(reference_samples);
    assert!(std::f64::consts::PI / 2.0 >= theta && theta >= 0.0);
    let diff = samples.unsqueeze(1) - reference_samples.unsqueeze(0);
    let metric = -((diff * theta).cos().log()).mean_dim([2].as_slice(), false, None::<Kind>);
    assert!(
        metric.isnan().any().int64_value(&[]) == 0,
        "if there's nan, try to reduce theta"
    );
    metric
}

/// Negative Chebyshev-style metric.
pub fn metric_neg_chebyshev(samples: &Tensor, reference_samples: &Tensor) -> Tensor {
    check_samples(samples);
    check_samples(reference_samples);
    let diff = samples.unsqueeze(1) - reference_samples.unsqueeze(0);
    diff.norm_scalaropt_dim(2.0, [-1].as_slice(), false)
        .min_dim(-1, false)
        .0
}

/// Cossin plus Chebyshev-style metric.
pub fn metric_neg_cossin_chebyshev(
    samples: &Tensor,
    reference_samples: &Tensor,
    theta: f64,
) -> Tensor {
    metric_neg_log_cos_sin(samples, reference_samples, theta)
        .min_dim(-1, false)
        .0
}
