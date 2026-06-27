//! Small eigen-decomposition helpers.

use tch::{Device, Kind, Tensor};

/// Generate a random Hermitian matrix.
pub fn rand_hermitian_matrix(dim: i64, device: Device) -> Tensor {
    let h = Tensor::randn([dim, dim], (Kind::ComplexFloat, device));
    &h + h.conj().transpose(0, 1)
}

/// Generate a random real symmetric matrix.
pub fn rand_real_symmetric_matrix(dim: i64, device: Device) -> Tensor {
    let mat = Tensor::randn([dim, dim], (Kind::Float, device));
    (&mat + mat.transpose(0, 1)) / 2.0
}
