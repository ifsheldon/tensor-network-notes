//! Trainable network modules.
//!
//! The Rust port keeps trainable models separate from `MPS`, which remains tensor-like and does not own an `nn::VarStore`.
