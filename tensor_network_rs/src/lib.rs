pub mod constants;
pub mod utils;

#[cfg(test)]
mod tests {
    use crate::utils::{allclose, set_seed};
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_allclose_basic() {
        let a = Tensor::f_from_slice(&[1.0_f64, 2.0, 3.0]).unwrap();
        let b = &a + 1e-7;
        assert!(allclose(&a, &b, None, None, false).unwrap());
        let c = &a + 1e-3;
        assert!(!allclose(&a, &c, None, Some(1e-8), false).unwrap());
    }

    #[test]
    fn test_set_seed_reproducible() {
        set_seed(42);
        let x1 = Tensor::randn([2, 3], (Kind::Float, Device::Cpu));
        set_seed(42);
        let x2 = Tensor::randn([2, 3], (Kind::Float, Device::Cpu));
        assert!(allclose(&x1, &x2, None, None, false).unwrap());
    }
}
