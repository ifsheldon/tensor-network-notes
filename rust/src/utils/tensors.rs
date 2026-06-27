//! Generic tensor helpers.

use tch::{Device, IndexOp, Kind, Tensor};

/// Create an identity tensor with equal dimensions in every mode.
pub fn identity_tensor(order: i64, dim: i64, kind: Kind, device: Device) -> Tensor {
    assert!(order > 0, "order must be positive");
    assert!(dim > 0, "dim must be positive");
    let tensor = Tensor::zeros(vec![dim; order as usize], (kind, device));
    for i in 0..dim {
        let mut view = tensor.shallow_clone();
        for _ in 0..order {
            view = view.i(i);
        }
        let mut view = view;
        let _ = view.fill_(1.0);
    }
    tensor
}

/// Compute the outer product of a non-empty list of vectors.
pub fn outer_product(vectors: &[Tensor]) -> Tensor {
    assert!(
        vectors.len() >= 2,
        "At least two vectors are required for outer product"
    );
    for (idx, vector) in vectors.iter().enumerate() {
        assert_eq!(
            vector.dim(),
            1,
            "Expected 1D tensor, got {}D tensor at index {idx}",
            vector.dim()
        );
    }
    let mut result = vectors[0].shallow_clone();
    for vector in vectors.iter().skip(1) {
        let result_dim = result.dim() as i64;
        result = result.unsqueeze(result_dim) * vector;
    }
    result
}

/// Return the zero computational-basis state as a rank-`num_qubits` tensor.
pub fn zeros_state(num_qubits: i64, kind: Kind, device: Device) -> Tensor {
    assert!(num_qubits > 0, "num_qubits must be positive");
    assert!(
        matches!(kind, Kind::ComplexFloat | Kind::ComplexDouble),
        "dtype must be complex64 or complex128"
    );
    let state = Tensor::zeros([2_i64.pow(num_qubits as u32)], (kind, device));
    let _ = state.i(0).fill_(1.0);
    state.reshape(vec![2; num_qubits as usize])
}

/// Normalize a tensor to `[0, 1]`, either globally or along one dimension.
pub fn normalize_tensor(tensor: &Tensor, dim: Option<i64>) -> Tensor {
    let (min_val, max_val) = match dim {
        None => (tensor.min(), tensor.max()),
        Some(dim) => {
            let (min_val, _) = tensor.min_dim(dim, true);
            let (max_val, _) = tensor.max_dim(dim, true);
            (min_val, max_val)
        }
    };
    (tensor - &min_val) / (max_val - min_val)
}

/// Rescale a tensor to a target range.
pub fn rescale_tensor(tensor: &Tensor, min_val: f64, max_val: f64, dim: Option<i64>) -> Tensor {
    normalize_tensor(tensor, dim) * (max_val - min_val) + min_val
}

/// Compute a simple sequential tensordot contraction over neighboring virtual bonds.
pub fn chain_tensordot(tensors: &[Tensor]) -> Tensor {
    assert!(
        tensors.len() >= 2,
        "At least two tensors are needed for contraction"
    );
    let mut result = tensors[0].shallow_clone();
    for tensor in tensors.iter().skip(1) {
        let dim = result.dim() as i64 - 1;
        result = result.tensordot(tensor, [dim], [0]);
    }
    result
}
