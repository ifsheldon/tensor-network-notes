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

    #[test]
    fn test_einsum() {
        let einsum_path = None::<Vec<i64>>; // always use default path, no manual path
        let a = Tensor::f_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
        let b = Tensor::f_from_slice(&[4.0_f64, 5.0, 6.0]).unwrap();
        let c = Tensor::einsum("a,b->", &[a, b], einsum_path);
        println!("{:?}", c);
    }

    // Note: tch's global RNG and test parallelism can interact; omit strict reproducibility test.
}
