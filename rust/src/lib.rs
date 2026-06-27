#![warn(missing_docs)]
#![warn(rustdoc::broken_intra_doc_links)]

//! # Tensor Network Code
//!
//! Rust experiments for the tensor-network notes.
//!
//! This crate is intentionally small for now. The first goal is to validate how rustdoc behaves
//! for the documentation patterns we will need while oxidizing the Python notebook code with
//! `tch-rs`.
//!
//! ## Experiment 1: Embed Images
//!
//! Rustdoc accepts normal Markdown image syntax. These examples intentionally use the same style
//! as the notebooks:
//!
//! ```markdown
//! ![MPS tensor](images/mps_example.png)
//! ![Tensor network examples](images/tensor_network_examples.png)
//! ![QFT example](images/qft_example.png)
//! ```
//!
//! For local rustdoc output, the referenced files must exist relative to the generated crate page,
//! such as `target/doc/tensor_network_code/images`.
//!
//! ![MPS tensor](images/mps_example.png)
//!
//! ![Tensor network examples](images/tensor_network_examples.png)
//!
//! ![QFT example](images/qft_example.png)
//!
//! ## Experiment 2: Render Math Equations
//!
//! Rustdoc does not render TeX math by itself in this crate, so `.cargo/config.toml` injects a
//! small MathJax header with `--html-in-header`.
//!
//! Inline math example: $\langle \psi \mid \psi \rangle = 1$.
//!
//! Display math example:
//!
//! $$
//! |\psi\rangle = \sum_{i_1,\ldots,i_N} A_1^{i_1} A_2^{i_2} \cdots A_N^{i_N} |i_1,\ldots,i_N\rangle.
//! $$
//!
//! Build the docs with:
//!
//! ```shell
//! cargo doc --no-deps
//! ```

pub mod algorithms;
pub mod error;
pub mod feature_mapping;
pub mod mps;
pub mod quantum_state;
pub mod tensor_gates;
pub mod types;
pub mod utils;

/// Returns the Cargo package name used for the Rust crate.
///
/// This is intentionally simple while the Rust side is being bootstrapped.
pub fn package_name() -> &'static str {
    "tensor-network-code"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_name_matches_manifest_name() {
        assert_eq!(package_name(), env!("CARGO_PKG_NAME"));
    }
}
