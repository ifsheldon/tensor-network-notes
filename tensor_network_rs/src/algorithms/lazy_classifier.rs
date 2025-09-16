use crate::{
    algorithms::quantum_kernels::{
        metric_neg_chebyshev, metric_neg_cossin_chebyshev, metric_neg_log_cos_sin,
    },
    utils::types::*,
};
use std::collections::BTreeSet;
use tch::{Kind, Tensor};

fn unique_sorted_labels(labels: &Tensor) -> Vec<UIdx> {
    let cpu = labels.to_kind(Kind::Int64).to_device(tch::Device::Cpu);
    let v: Vec<TInt> = Vec::<TInt>::try_from(cpu).expect("to vec TInt");
    let mut set = BTreeSet::new();
    for x in v {
        set.insert(x);
    }
    set.into_iter().cast_items().collect()
}

fn euclidean_mean(samples: &Tensor, refs: &Tensor) -> Tensor {
    let x2 = samples
        .square()
        .sum_dim_intlist([-1].as_slice(), false, samples.kind())
        .unsqueeze(1); // [n,1]
    let y2 = refs
        .square()
        .sum_dim_intlist([-1].as_slice(), false, refs.kind())
        .unsqueeze(0); // [1,m]
    let dot = samples.matmul(&refs.transpose(0, 1)); // [n,m]
    let d2: Tensor = x2 + y2 - 2.0 * dot;
    let d = d2.clamp_min(0.0).sqrt();
    d.mean_dim([-1].as_slice(), false, samples.kind()) // [n]
}

pub enum KernelKind {
    Euclidean,
    NllCossin { theta: f64 },
    RNllCossin { theta: f64, beta: f64 },
    Chebyshev,
    CossinChebyshev { theta: f64 },
}

pub fn lazy_classify(
    samples: &Tensor,           // [N, F]
    reference_samples: &Tensor, // [M, F]
    reference_labels: &Tensor,  // [M]
    kernel: KernelKind,
) -> Tensor {
    assert!(samples.dim() == 2 && reference_samples.dim() == 2);
    let classes = unique_sorted_labels(reference_labels);
    let mut dists: Vec<Tensor> = Vec::with_capacity(classes.len());
    for c in &classes {
        let mask = reference_labels.eq(c.to_tint());
        let idx = mask.nonzero().squeeze_dim(1);
        let refs_c = reference_samples.f_index_select(0, &idx).unwrap();
        let dist_c = match kernel {
            KernelKind::Euclidean => euclidean_mean(samples, &refs_c),
            KernelKind::NllCossin { theta } => {
                let mat = metric_neg_log_cos_sin(samples, &refs_c, theta, None);
                mat.mean_dim([-1].as_slice(), false, samples.kind())
            }
            KernelKind::RNllCossin { theta, beta } => {
                let mat = metric_neg_log_cos_sin(samples, &refs_c, theta, None);
                (mat * beta)
                    .exp()
                    .mean_dim([-1].as_slice(), false, samples.kind())
            }
            KernelKind::Chebyshev => metric_neg_chebyshev(samples, &refs_c, None),
            KernelKind::CossinChebyshev { theta } => {
                metric_neg_cossin_chebyshev(samples, &refs_c, theta, None)
            }
        };
        dists.push(dist_c);
    }
    let prob = Tensor::stack(&dists, 1); // [N, C]
    let pred_idx = prob
        .argmin(1, false)
        .to_kind(Kind::Int64)
        .to_device(tch::Device::Cpu); // [N]
    // Map back to class ids without tensor indexing to avoid dtype/device pitfalls
    let idx_vec: Vec<TInt> = Vec::<TInt>::try_from(pred_idx).unwrap();
    let mapped: Vec<TInt> = idx_vec
        .into_iter()
        .map(|ix| classes[ix as usize].to_tint())
        .collect();
    Tensor::from_slice(&mapped).to_kind(Kind::Int64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_lazy_classifier_euclidean() {
        // Two clusters in 2D: class 0 near (0,0), class 1 near (1,1)
        let dev = Device::Cpu;
        let k = Kind::Float;
        let refs = Tensor::from_slice(&[0.0, 0.0, 1.0, 1.0])
            .view([2, 2])
            .to_kind(k)
            .to_device(dev);
        let labels = Tensor::from_slice(&[0, 1]).to_kind(Kind::Int64);
        let samples = Tensor::from_slice(&[0.05, 0.0, 0.9, 0.95, 0.1, 0.2, 0.8, 1.0])
            .view([4, 2])
            .to_kind(k);
        let pred = lazy_classify(&samples, &refs, &labels, KernelKind::Euclidean);
        let v: Vec<TInt> = Vec::<TInt>::try_from(pred).unwrap();
        assert_eq!(v, vec![0, 1, 0, 1]);
    }
}
