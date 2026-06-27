//! Error types for fallible Tensor Network operations.

/// Result alias used by fallible public APIs.
pub type Result<T> = std::result::Result<T, TensorNetworkError>;

/// Errors produced by IO and other recoverable runtime operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorNetworkError {
    /// Error propagated from `tch`.
    #[error(transparent)]
    Tch(#[from] tch::TchError),

    /// Error propagated from filesystem IO.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// A safetensors file missed an expected tensor entry.
    #[error("missing tensor key {0:?}")]
    MissingTensorKey(String),

    /// A safetensors file had an invalid tensor key.
    #[error("invalid tensor key {0:?}")]
    InvalidTensorKey(String),

    /// A dataset or artifact path was unavailable.
    #[error("artifact is unavailable: {0}")]
    MissingArtifact(String),
}
