//! Constants shared across the crate.
//! These match PyTorch defaults to ease parity with the Python notebooks.

use crate::utils::types::TInt;
use tch::Kind;

/// Relative tolerance used for approximate comparisons.
/// Matches PyTorch's default `rtol=1e-5` in `torch.allclose` and `torch.isclose`.
pub const RTOL_DEFAULT: f64 = 1e-5;

/// Absolute tolerance used for approximate comparisons.
/// Matches PyTorch's default `atol=1e-8` in `torch.allclose` and `torch.isclose`.
pub const ATOL_DEFAULT: f64 = 1e-8;

/// Default behavior for NaN equality in closeness checks (PyTorch: `equal_nan=false`).
pub const EQUAL_NAN_DEFAULT: bool = false;

pub const NO_OPT_PATH: Option<Vec<TInt>> = None;

pub const DEFAULT_FLOAT_KIND: Kind = Kind::Float;
