//! Algorithm modules.

pub mod calc_ground_state;
pub mod eigen_decomposition;
pub mod feature_selection;
pub mod gmps;
pub mod imaginary_time_evolution;
pub mod lazy_classifier;
pub mod quantum_kernels;
pub mod tensor_decomposition;
pub mod time_evolving_block_decimation;

pub use calc_ground_state::*;
pub use eigen_decomposition::*;
pub use feature_selection::*;
pub use gmps::*;
pub use imaginary_time_evolution::*;
pub use lazy_classifier::*;
pub use quantum_kernels::*;
pub use tensor_decomposition::*;
pub use time_evolving_block_decimation::*;
