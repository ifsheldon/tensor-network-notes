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
        !vectors.is_empty(),
        "At least one vector is required for outer product"
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

/// Contract tensors with a named-axis expression and explicit shared-dimension aliases.
pub fn tensor_contract(
    tensors: &[&Tensor],
    pattern: &str,
    shared: impl einops::SharedDims,
) -> Tensor {
    assert!(
        tensors.len() >= 2,
        "At least two tensors are needed for contraction"
    );
    let equation = einops::contract_str(pattern, shared);
    Tensor::einsum(&equation, tensors, None::<i64>)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &Tensor, expected: &Tensor) {
        assert!(
            actual.allclose(expected, 1e-5, 1e-8, false),
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn identity_tensor_marks_only_matching_indices() {
        let tensor = identity_tensor(3, 2, Kind::Float, Device::Cpu);
        let expected =
            Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).reshape([2, 2, 2]);
        assert_close(&tensor, &expected);
    }

    #[test]
    fn zeros_state_returns_zero_computational_basis_state() {
        let state = zeros_state(2, Kind::ComplexFloat, Device::Cpu);
        assert_eq!(state.size(), vec![2, 2]);
        let expected = Tensor::from_slice(&[1.0_f32, 0.0, 0.0, 0.0]).reshape([2, 2]);
        assert_close(&state.abs(), &expected);
    }

    #[test]
    fn normalize_tensor_can_normalize_globally() {
        let tensor = Tensor::from_slice(&[2.0_f32, 4.0, 6.0]);
        let normalized = normalize_tensor(&tensor, None);
        let expected = Tensor::from_slice(&[0.0_f32, 0.5, 1.0]);
        assert_close(&normalized, &expected);
    }

    #[test]
    fn normalize_tensor_can_normalize_along_dimension() {
        let tensor = Tensor::from_slice(&[1.0_f32, 3.0, 2.0, 6.0]).reshape([2, 2]);
        let normalized = normalize_tensor(&tensor, Some(1));
        let expected = Tensor::from_slice(&[0.0_f32, 1.0, 0.0, 1.0]).reshape([2, 2]);
        assert_close(&normalized, &expected);
    }

    #[test]
    fn rescale_tensor_maps_to_requested_range() {
        let tensor = Tensor::from_slice(&[2.0_f32, 4.0, 6.0]);
        let rescaled = rescale_tensor(&tensor, -1.0, 1.0, None);
        let expected = Tensor::from_slice(&[-1.0_f32, 0.0, 1.0]);
        assert_close(&rescaled, &expected);
    }

    #[test]
    fn tensor_contract_matches_compact_einsum_for_one_shared_group() {
        let a = Tensor::arange(6, (Kind::Float, Device::Cpu)).reshape([2, 3]);
        let b = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([3, 4]);
        let actual = tensor_contract(
            &[&a, &b],
            "a_left a_right, b_left b_right -> a_left b_right",
            ["a_right", "b_left"],
        );
        let expected = Tensor::einsum("ab,bc->ac", &[&a, &b], None::<i64>);
        assert_close(&actual, &expected);
    }

    #[test]
    fn tensor_contract_matches_compact_einsum_for_multiple_shared_groups() {
        let a = Tensor::arange(30, (Kind::Float, Device::Cpu)).reshape([2, 3, 5]);
        let b = Tensor::arange(60, (Kind::Float, Device::Cpu)).reshape([3, 4, 5]);
        let actual = tensor_contract(
            &[&a, &b],
            "a_left shared_one x, b_left b_right y -> a_left b_right",
            [["shared_one", "b_left"], ["x", "y"]],
        );
        let expected = Tensor::einsum("abc,bdc->ad", &[&a, &b], None::<i64>);
        assert_close(&actual, &expected);
    }

    #[test]
    fn tensor_contract_supports_scalar_output() {
        let a = Tensor::arange(6, (Kind::Float, Device::Cpu)).reshape([2, 3]);
        let b = Tensor::arange(6, (Kind::Float, Device::Cpu)).reshape([3, 2]);
        let actual = tensor_contract(
            &[&a, &b],
            "a_left a_right, b_left b_right ->",
            [["a_right", "b_left"], ["a_left", "b_right"]],
        );
        let expected = Tensor::einsum("ab,ba->", &[&a, &b], None::<i64>);
        assert_close(&actual, &expected);
    }

    #[test]
    fn tensor_contract_allows_output_alias_names() {
        let a = Tensor::arange(6, (Kind::Float, Device::Cpu)).reshape([2, 3]);
        let b = Tensor::arange(12, (Kind::Float, Device::Cpu)).reshape([3, 4]);
        let actual = tensor_contract(
            &[&a, &b],
            "a_left a_right, b_left b_right -> a_left a_right",
            ["b_left", "a_right"],
        );
        let expected = Tensor::einsum("ab,bc->ab", &[&a, &b], None::<i64>);
        assert_close(&actual, &expected);
    }
}
