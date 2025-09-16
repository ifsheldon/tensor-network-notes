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
        _ => panic!("complex_kind must be ComplexFloat or ComplexDouble"),
    };
    let re = Tensor::from_slice(values_re).to_kind(real_kind).view(shape);
    let im = Tensor::from_slice(values_im).to_kind(real_kind).view(shape);
    Tensor::complex(&re, &im)
}

pub fn enable_efficient_mode() -> bool {
    std::env::var("TENSOR_NETWORK_SLOW_MODE").is_err()
}
