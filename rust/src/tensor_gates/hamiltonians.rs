//! Hamiltonian builders.

use tch::{Device, Kind, Tensor};

use crate::tensor_gates::functional::{Pauli, pauli_operator};

/// Calculate the two-site Heisenberg Hamiltonian.
pub fn heisenberg(
    jx: f64,
    jy: f64,
    jz: f64,
    double_precision: bool,
    return_matrix: bool,
    device: Device,
) -> Tensor {
    let kind = if double_precision {
        Kind::Double
    } else {
        Kind::Float
    };
    let pauli_x = pauli_operator(Pauli::X, double_precision, false, device).to_kind(kind);
    let pauli_y = pauli_operator(Pauli::Y, double_precision, false, device).to_kind(kind);
    let pauli_z = pauli_operator(Pauli::Z, double_precision, false, device).to_kind(kind);
    let h = Tensor::einsum("ab,ij->aibj", &[&pauli_x, &pauli_x], None::<i64>) * jx
        + Tensor::einsum("ab,ij->aibj", &[&pauli_y, &pauli_y], None::<i64>).real() * jy
        + Tensor::einsum("ab,ij->aibj", &[&pauli_z, &pauli_z], None::<i64>) * jz;
    let h = h / 4.0;
    if return_matrix { h.reshape([4, 4]) } else { h }
}
