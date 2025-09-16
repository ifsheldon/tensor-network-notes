use crate::types::*;
use tch::{IndexOp, Tensor};

fn check_samples(samples: &Tensor) {
    assert!(samples.dim() == 2);
    let min = samples.min();
    let max = samples.max();
    assert!(min.double_value(&[]) >= 0.0);
    assert!(max.double_value(&[]) <= 1.0);
}

pub fn metric_matrix_neg_log_cos_sin(samples: &Tensor, theta: f64, dedup: bool) -> Tensor {
    check_samples(samples);
    assert!((0.0..=std::f64::consts::FRAC_PI_2).contains(&theta));
    let n = samples.size()[0];
    if dedup {
        let metric = Tensor::zeros([n, n], (samples.kind(), samples.device()));
        for i in 0..(n - 1) {
            let a = samples.i(i).view([1, -1]);
            let others = samples.i((i + 1)..);
            let diff = &a - &others;
            let distances = (-(diff * theta).cos().log()).mean_dim(-1, false, samples.kind());
            assert!(
                distances.isnan().any().int64_value(&[]) == 0,
                "nan in distances; try reducing theta"
            );
            metric.i((i, (i + 1)..)).copy_(&distances);
            metric.i((((i + 1)..), i)).copy_(&distances);
        }
        metric
    } else {
        let a = samples.unsqueeze(0);
        let b = samples.unsqueeze(1);
        let diff = &a - &b;
        let m = (-(diff * theta).cos().log()).mean_dim(-1, false, samples.kind());
        assert!(
            m.isnan().any().int64_value(&[]) == 0,
            "nan in metric; try reducing theta"
        );
        for i in 0..n {
            let _ = m.i((i, i)).fill_(0.0);
        }
        m
    }
}

pub fn metric_matrix_neg_log_cos_sin_method(
    samples: &Tensor,
    theta: f64,
    calculation_method: &str,
) -> Tensor {
    match calculation_method {
        "deduplicate" => metric_matrix_neg_log_cos_sin(samples, theta, true),
        "no_deduplicate" => metric_matrix_neg_log_cos_sin(samples, theta, false),
        _ => panic!("calculation_method must be 'deduplicate' or 'no_deduplicate'"),
    }
}

pub fn metric_neg_log_cos_sin(
    samples: &Tensor,
    refs: &Tensor,
    theta: f64,
    batch_size: Option<Num>,
) -> Tensor {
    check_samples(samples);
    check_samples(refs);
    assert!((0.0..=std::f64::consts::FRAC_PI_2).contains(&theta));
    let n = samples.size()[0].cast();
    let bs = batch_size.unwrap_or(if n >= 2 { n / 2 } else { 1 });
    let mut out: Vec<Tensor> = Vec::new();
    let mut start = 0;
    let refs_u = refs.unsqueeze(0);
    while start < n {
        let end = (start + bs).min(n);
        let batch = samples.i(start.to_tint()..end.to_tint()).unsqueeze(1);
        let diff = &batch - &refs_u;
        let res = (-(diff * theta).cos().log()).mean_dim(-1, false, samples.kind());
        assert!(
            res.isnan().any().int64_value(&[]) == 0,
            "nan in metric; try reducing theta"
        );
        out.push(res);
        start = end;
    }
    Tensor::cat(&out, 0)
}

pub fn metric_neg_chebyshev(samples: &Tensor, refs: &Tensor, batch_size: Option<Num>) -> Tensor {
    check_samples(samples);
    check_samples(refs);
    let n = samples.size()[0].cast();
    let bs = batch_size.unwrap_or(if n >= 2 { n / 2 } else { 1 });
    let mut out: Vec<Tensor> = Vec::new();
    let mut start = 0;
    let refs_u = refs.unsqueeze(0);
    while start < n {
        let end = (start + bs).min(n);
        let batch = samples.i(start.to_tint()..end.to_tint()).unsqueeze(1);
        let diff = &batch - &refs_u;
        let res = diff
            .norm_scalaropt_dim(2.0, [-1].as_slice(), false)
            .min_dim(-1, false)
            .0;
        out.push(res);
        start = end;
    }
    Tensor::cat(&out, 0)
}

pub fn metric_neg_cossin_chebyshev(
    samples: &Tensor,
    refs: &Tensor,
    theta: f64,
    batch_size: Option<Num>,
) -> Tensor {
    check_samples(samples);
    check_samples(refs);
    let n = samples.size()[0].cast();
    let bs = batch_size.unwrap_or(if n >= 2 { n / 2 } else { 1 });
    let mut out: Vec<Tensor> = Vec::new();
    let mut start = 0;
    let refs_u = refs.unsqueeze(0);
    while start < n {
        let end = (start + bs).min(n);
        let batch = samples.i(start.to_tint()..end.to_tint()).unsqueeze(1);
        let diff = &batch - &refs_u;
        let mut res = (-(diff * theta).cos().log()).mean_dim(-1, false, samples.kind());
        assert!(
            res.isnan().any().int64_value(&[]) == 0,
            "nan in metric; try reducing theta"
        );
        res = res.min_dim(-1, false).0;
        out.push(res);
        start = end;
    }
    Tensor::cat(&out, 0)
}
