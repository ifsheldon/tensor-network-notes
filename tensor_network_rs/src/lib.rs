#[cfg(test)]
mod tests {
    use tch::nn::{Adam, OptimizerConfig, VarStore};
    use tch::{Device, Kind, Tensor};

    #[test]
    pub fn it_works() {
        let a = Tensor::ones([2, 2], (Kind::Float, Device::Cpu));
        let b = a * 2. * 2.;
        let result = b.sum(Kind::Float).double_value(&[]);
        assert_eq!(result, 2. * 2. * (2. * 2.));
    }

    #[test]
    pub fn test_optimizer() {
        let vs = VarStore::new(Device::Cpu);
        let root = vs.root();
        let mut adam = Adam::default().build(&vs, 1e-1).unwrap();
        let v = root.randn("v", &[2, 2, 2, 2], 0.0, 1.0);
        let v_copy = v.copy();
        v_copy.print();
        let loss = v.pow_tensor_scalar(2).sum(Kind::Float);
        loss.backward();
        let gradient = v.grad();
        assert!(gradient.allclose(&(2 * v.copy()), 1e-5, 1e-5, false));
        adam.step();
        adam.zero_grad();
        v.print();
        assert!(!v.allclose(&v_copy, 1e-5, 1e-5, false));
    }
}
