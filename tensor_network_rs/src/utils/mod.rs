pub mod checking;
pub mod constants;
pub mod einsum;
pub mod mapping;
pub mod types;

pub use checking::*;
pub use constants::*;
pub use einsum::*;
pub use mapping::*;
pub use types::*;

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

/// Build a complex tensor from real/imag slices and a target shape.
/// `complex_kind` must be `Kind::ComplexFloat` or `Kind::ComplexDouble`.
pub fn complex_from_slices(
    values_re: &[f64],
    values_im: &[f64],
    shape: &[i64],
    complex_kind: Kind,
) -> Tensor {
    assert!(matches!(
        complex_kind,
        Kind::ComplexFloat | Kind::ComplexDouble
    ));
    assert_eq!(values_re.len(), values_im.len());
    let real_kind = match complex_kind {
        Kind::ComplexFloat => Kind::Float,
        Kind::ComplexDouble => Kind::Double,
        _ => unreachable!(),
    };
    let re = Tensor::f_from_slice(values_re)
        .unwrap()
        .to_kind(real_kind)
        .view(shape);
    let im = Tensor::f_from_slice(values_im)
        .unwrap()
        .to_kind(real_kind)
        .view(shape);
    Tensor::f_complex(&re, &im).unwrap()
}
