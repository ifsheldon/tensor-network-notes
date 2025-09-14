pub mod constants;
pub mod eigen_decomposition;
pub mod tensor_gates {
    pub mod functional;
}
pub mod utils;

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

    // Note: tch's global RNG and test parallelism can interact; omit strict reproducibility test.
}
