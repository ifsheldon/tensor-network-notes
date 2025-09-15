use tch::{IndexOp, Tensor};

pub fn cossin_feature_map(samples: &Tensor, theta: f64, check_range: bool) -> Tensor {
    let mut x = samples.shallow_clone();
    if x.dim() == 1 {
        x = x.unsqueeze(0);
    }
    if check_range {
        let min_ok = x.min().double_value(&[]) >= 0.0;
        let max_ok = x.max().double_value(&[]) <= 1.0;
        assert!(
            min_ok && max_ok,
            "samples must be in [0,1] or set check_range=false"
        );
    }
    let angle = &x * (theta * std::f64::consts::PI);
    let cos = angle.cos();
    let sin = angle.sin();
    Tensor::stack(&[cos, sin], -1)
}

pub fn feature_map_to_qubit_state(features: &Tensor) -> Tensor {
    assert!(features.dim() == 3 && features.size()[2] == 2);
    let f = features.size()[1];
    // split along feature axis and contract via named_einsum with symbolic labels
    let mut parts: Vec<Tensor> = Vec::with_capacity(f as usize);
    for i in 0..f {
        parts.push(features.i((.., i, ..))); // [B,2]
    }
    // Build equation like "B f0, B f1, ... -> B f0 f1 ..." using named_einsum
    let inputs: Vec<String> = (0..f).map(|i| format!("B f{}", i)).collect();
    let outputs: Vec<String> = (0..f).map(|i| format!("f{}", i)).collect();
    let eq = format!("{} -> B {}", inputs.join(","), outputs.join(" "));
    crate::utils::einsum::named_einsum(&eq, &parts)
}

pub fn linear_mapping(samples: &Tensor) -> Tensor {
    let mut x = samples.shallow_clone();
    if x.dim() == 1 {
        x = x.reshape([1, -1]);
    }
    let one = Tensor::from(1.0).to_kind(x.kind()).to_device(x.device());
    let comp = &one - &x;
    Tensor::stack(&[x, comp], -1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;
    #[test]
    fn test_cossin_shapes() {
        let s = Tensor::f_from_slice(&[0.0, 0.5, 1.0])
            .unwrap()
            .to_kind(Kind::Float)
            .view([1, 3]);
        let f = cossin_feature_map(&s, 1.0, true);
        assert_eq!(f.size(), vec![1, 3, 2]);
    }
    #[test]
    fn test_qubit_state_shape() {
        let f = Tensor::ones([2, 4, 2], (Kind::Float, tch::Device::Cpu));
        let q = feature_map_to_qubit_state(&f);
        assert_eq!(q.size(), vec![2, 2, 2, 2, 2]);
    }
}
