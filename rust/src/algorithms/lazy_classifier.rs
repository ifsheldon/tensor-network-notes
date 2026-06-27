//! Lazy classifier helpers.

use tch::{Kind, Tensor};

use crate::algorithms::quantum_kernels::{
    metric_neg_chebyshev, metric_neg_cossin_chebyshev, metric_neg_log_cos_sin,
};
use crate::types::Kernel;

/// Classify samples by comparing against labeled reference samples.
pub fn lazy_classify(
    samples: &Tensor,
    reference_samples: &Tensor,
    reference_labels: &Tensor,
    kernel: Kernel,
    theta: f64,
    beta: Option<f64>,
) -> Tensor {
    assert_eq!(reference_labels.dim(), 1);
    assert_eq!(reference_samples.dim(), 2);
    assert_eq!(samples.dim(), 2);
    let classes = reference_labels.unique_dim(0, true, false, false).0;
    let mut probs = Vec::new();
    for idx in 0..classes.size()[0] {
        let class = classes.get(idx);
        let mask = reference_labels.eq_tensor(&class);
        let ref_c = reference_samples.index(&[Some(&mask)]);
        let dist = match kernel {
            Kernel::Euclidean => Tensor::cdist(samples, &ref_c, 2.0, None::<i64>).mean_dim(
                [1].as_slice(),
                false,
                None::<Kind>,
            ),
            Kernel::NllCossin => metric_neg_log_cos_sin(samples, &ref_c, theta).mean_dim(
                [1].as_slice(),
                false,
                None::<Kind>,
            ),
            Kernel::ReweightedNllCossin => {
                let beta = beta.expect("beta is required for r_nll_cossin");
                let dist = metric_neg_log_cos_sin(samples, &ref_c, theta);
                Tensor::from(beta)
                    .pow(&dist)
                    .mean_dim([1].as_slice(), false, None::<Kind>)
            }
            Kernel::Chebyshev => metric_neg_chebyshev(samples, &ref_c),
            Kernel::CossinChebyshev => metric_neg_cossin_chebyshev(samples, &ref_c, theta),
        };
        probs.push(dist);
    }
    let probs = Tensor::stack(&probs, 1);
    let pred = probs.argmin(1, false);
    classes.index_select(0, &pred)
}
