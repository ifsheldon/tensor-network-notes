pub mod constants;
pub mod eigen_decomposition;
pub mod tensor_gates {
    pub mod functional;
}
pub mod quantum_state {
    pub mod functional;
}
pub mod mps {
    pub mod functional;
    pub mod modules;
}
pub mod algorithms {
    pub mod gmps;
    pub mod imaginary_time_evolution;
    pub mod lazy_classifier;
    pub mod quantum_kernels;
    pub mod tensor_decomposition;
    pub mod time_evolving_block_decimation;
}
pub mod utils;
pub mod feature_mapping;

#[cfg(test)]
mod tests {
    use crate::utils::allclose;
    use tch::Tensor;

    #[test]
    fn test_allclose_basic() {
        let a = Tensor::f_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
        let b = &a + 1e-7;
        assert!(allclose(&a, &b, None, None, false).unwrap());
        let c = &a + 1e-3;
        assert!(!allclose(&a, &c, None, Some(1e-8), false).unwrap());
    }

    #[test]
    fn test_einsum() {
        let einsum_path = None::<Vec<i64>>; // always use default path, no manual path
        let a = Tensor::f_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
        let b = Tensor::f_from_slice(&[4.0_f64, 5.0, 6.0]).unwrap();
        let c = Tensor::einsum("a,b->", &[a, b], einsum_path);
        println!("{:?}", c);
    }

    // Note: tch's global RNG and test parallelism can interact; omit strict reproducibility test.
    #[test]
    fn test_gmps_selected_features_degenerate_to_full() {
        use crate::mps::modules::{MPS, MPSType};
        use tch::{Device, Kind, Tensor};
        let dev = Device::Cpu;
        let k = Kind::Float;
        let length = 3;
        let phys = 2;
        let virt = 2;
        let m = MPS::random(length, phys, virt, MPSType::Open, k, dev, false);
        let samples = Tensor::rand([5, length, phys], (k, dev));
        let full = crate::algorithms::gmps::eval_nll(&samples, &m, false);
        let idx: Vec<i64> = (0..length as i64).collect();
        let sub = crate::algorithms::gmps::eval_nll_selected_features(&samples, &m, &idx, false);
        let diff = (full - sub).abs().max();
        let d = diff.double_value(&[]);
        assert!(d < 1e-6, "subset(all) should equal full, got {}", d);
    }
}
