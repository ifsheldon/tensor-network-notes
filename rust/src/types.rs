//! Small typed options shared across the Rust port.

/// Preferred device selection.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DevicePreference {
    /// Prefer CUDA, then MPS, then CPU.
    Auto,
    /// Require CUDA.
    Cuda,
    /// Require MPS.
    Mps,
    /// Use CPU.
    Cpu,
}

/// Orthogonalization algorithm.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OrthogonalizationMode {
    /// Singular-value decomposition.
    Svd,
    /// QR decomposition.
    Qr,
}

/// Quantum-circuit gate layout.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum GatePattern {
    /// Brick-wall layout.
    Brick,
    /// Staircase layout.
    Stair,
}

/// Lazy-classifier kernel.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Kernel {
    /// Euclidean distance.
    Euclidean,
    /// Negative log cossin distance.
    NllCossin,
    /// Reweighted negative log cossin distance.
    ReweightedNllCossin,
    /// Chebyshev-style distance.
    Chebyshev,
    /// Cossin plus Chebyshev-style distance.
    CossinChebyshev,
}

/// Matrix-product-state boundary condition.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MPSType {
    /// Open-boundary MPS.
    Open,
    /// Periodic-boundary MPS.
    Periodic,
}

impl MPSType {
    /// Determine the MPS type from local tensors.
    pub fn from_tensors(tensors: &[tch::Tensor]) -> Self {
        assert!(!tensors.is_empty(), "MPS must have at least one tensor");
        let first = tensors.first().expect("checked non-empty").size();
        let last = tensors.last().expect("checked non-empty").size();
        assert_eq!(first.len(), 3, "MPS local tensors must be rank-3");
        assert_eq!(last.len(), 3, "MPS local tensors must be rank-3");
        if first[0] == 1 && last[2] == 1 {
            Self::Open
        } else {
            Self::Periodic
        }
    }
}
