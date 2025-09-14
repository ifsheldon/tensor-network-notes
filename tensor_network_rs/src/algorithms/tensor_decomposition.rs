use tch::Tensor;

pub fn make_matrix(tensor: &Tensor, left_index: i64) -> Tensor {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    assert!(left_index >= 0 && left_index < order);
    let t = tensor.movedim(left_index, 0);
    t.reshape([t.size()[0], -1])
}

pub fn tucker_decomposition(tensor: &Tensor) -> (Tensor, Vec<Tensor>, Vec<i64>) {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    let mut matrices_u: Vec<Tensor> = Vec::with_capacity(order as usize);
    let mut ranks: Vec<i64> = Vec::with_capacity(order as usize);
    for i in 0..(order as usize) {
        let m = make_matrix(tensor, i as i64);
        let (u, s, _vh) = m.svd(true, false);
        matrices_u.push(u);
        // Simple threshold
        let rank = s.gt(1e-10).sum(s.kind()).int64_value(&[]);
        ranks.push(rank);
    }
    let mut core = tensor.shallow_clone();
    for (i, u) in matrices_u.iter().enumerate().take(order as usize) {
        core = core.tensordot(&u.conj(), [i as i64].as_slice(), [0_i64].as_slice());
    }
    (core, matrices_u, ranks)
}

pub fn reduced_matrix(core_tensor: &Tensor, n: i64) -> Tensor {
    let order = core_tensor.dim() as i64;
    assert!(order >= 2 && n >= 0 && n < order);
    let m = make_matrix(core_tensor, n);
    let mh = m.conj().transpose(0, 1);
    m.matmul(&mh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;
    #[test]
    fn test_make_matrix_shape() {
        let t = Tensor::arange(24, (tch::Kind::Float, tch::Device::Cpu)).view([2, 3, 4]);
        let m = make_matrix(&t, 1);
        assert_eq!(m.size(), vec![3, 8]);
    }
}
