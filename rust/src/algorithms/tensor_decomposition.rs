//! Tensor-decomposition helpers.

use tch::{Kind, Tensor};

use crate::utils::tensors::outer_product;

/// Convert a tensor into a matrix by moving `left_index` to the front.
pub fn make_matrix(tensor: &Tensor, left_index: i64) -> Tensor {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    assert!(0 <= left_index && left_index < order);
    tensor
        .movedim_int(left_index, 0)
        .reshape([tensor.size()[left_index as usize], -1])
}

/// Tucker decomposition using SVD on each mode.
pub fn tucker_decomposition(tensor: &Tensor) -> (Tensor, Vec<Tensor>, Vec<i64>) {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    let mut matrices_u = Vec::new();
    let mut ranks = Vec::new();
    for idx in 0..order {
        let matrix = make_matrix(tensor, idx);
        let (u, s, _) = Tensor::linalg_svd(&matrix, false, "");
        let rank = s
            .ne(0.0)
            .to_kind(Kind::Int64)
            .sum(None::<Kind>)
            .int64_value(&[]);
        matrices_u.push(u);
        ranks.push(rank);
    }
    let mut core = tensor.shallow_clone();
    for matrix in &matrices_u {
        core = core.tensordot(&matrix.conj(), [0], [0]);
    }
    (core, matrices_u, ranks)
}

/// Reduced matrix of a Tucker core.
pub fn reduced_matrix(core_tensor: &Tensor, n: i64) -> Tensor {
    let order = core_tensor.dim() as i64;
    assert!(order >= 2);
    assert!(0 <= n && n < order);
    let matrix = make_matrix(core_tensor, n);
    matrix.matmul(&matrix.conj().transpose(0, 1))
}

/// Rank-1 tensor from vectors.
pub fn rank1_tensor(vectors: &[Tensor]) -> Tensor {
    outer_product(vectors)
}
