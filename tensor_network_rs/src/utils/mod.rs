use crate::constants::{ATOL_DEFAULT, RTOL_DEFAULT};
use tch::{Kind, Tensor};

/// Set global random seed for tch/LibTorch operations.
pub fn set_seed(seed: i64) {
    tch::manual_seed(seed);
}

/// Checks whether two tensors are approximately equal using tolerances.
/// If `rtol`/`atol` are `None`, PyTorch's defaults are used.
pub fn allclose(
    a: &Tensor,
    b: &Tensor,
    rtol: Option<f64>,
    atol: Option<f64>,
    equal_nan: bool,
) -> tch::Result<bool> {
    let rtol = rtol.unwrap_or(RTOL_DEFAULT);
    let atol = atol.unwrap_or(ATOL_DEFAULT);

    // Broadcasting follows PyTorch rules via arithmetic ops.
    let diff = a - b;
    let abs_diff = diff.abs();
    let rhs = b.abs() * rtol + atol;
    let close = if equal_nan {
        let both_nan = a.isnan() * b.isnan();
        let tol_met = abs_diff.le_tensor(&rhs);
        (tol_met + both_nan).to_kind(Kind::Bool)
    } else {
        abs_diff.le_tensor(&rhs)
    };
    // Reduce to a single boolean
    let all = close.all();
    // Convert 0-dim bool tensor to bool
    Ok(all.int64_value(&[]) != 0)
}

/// Helper to create a tensor of zeros with same shape and kind as input.
pub fn zeros_like(x: &Tensor) -> Tensor {
    Tensor::zeros_like(x)
}

/// Returns a CPU float kind matching typical default dtype for math.
pub fn default_float_kind() -> Kind {
    Kind::Float
}

/// Convert a scalar or slice of f64 into a 1D tensor (CPU, Float64).
pub fn tensor1(values: &[f64]) -> Tensor {
    Tensor::f_from_slice(values).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allclose_nan_behavior() {
        let a = tensor1(&[f64::NAN, 1.0]);
        let b = tensor1(&[f64::NAN, 1.0 + 1e-9]);
        assert!(!allclose(&a, &b, None, None, false).unwrap());
        assert!(allclose(&a, &b, None, None, true).unwrap());
    }
}

pub mod checking;
pub mod mapping;
