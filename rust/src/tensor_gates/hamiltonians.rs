//! Hamiltonian builders.

use einops::einsumstr;
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
    let xx = Tensor::einsum(
        einsumstr!("out0 in0, out1 in1 -> out0 out1 in0 in1"),
        &[&pauli_x, &pauli_x],
        None::<i64>,
    );
    let yy = Tensor::einsum(
        einsumstr!("out0 in0, out1 in1 -> out0 out1 in0 in1"),
        &[&pauli_y, &pauli_y],
        None::<i64>,
    )
    .real();
    let zz = Tensor::einsum(
        einsumstr!("out0 in0, out1 in1 -> out0 out1 in0 in1"),
        &[&pauli_z, &pauli_z],
        None::<i64>,
    );
    let h = xx * jx + yy * jy + zz * jz;
    let h = h / 4.0;
    if return_matrix { h.reshape([4, 4]) } else { h }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_heisenberg_reference(
        jx: f64,
        jy: f64,
        jz: f64,
        double_precision: bool,
        return_matrix: bool,
    ) -> Tensor {
        let kind = if double_precision {
            Kind::Double
        } else {
            Kind::Float
        };
        let pauli_x = pauli_operator(Pauli::X, double_precision, false, Device::Cpu).to_kind(kind);
        let pauli_y = pauli_operator(Pauli::Y, double_precision, false, Device::Cpu).to_kind(kind);
        let pauli_z = pauli_operator(Pauli::Z, double_precision, false, Device::Cpu).to_kind(kind);
        let h = Tensor::einsum("ab,ij->aibj", &[&pauli_x, &pauli_x], None::<i64>) * jx
            + Tensor::einsum("ab,ij->aibj", &[&pauli_y, &pauli_y], None::<i64>).real() * jy
            + Tensor::einsum("ab,ij->aibj", &[&pauli_z, &pauli_z], None::<i64>) * jz;
        let h = h / 4.0;
        if return_matrix { h.reshape([4, 4]) } else { h }
    }

    #[test]
    fn heisenberg_matches_compact_reference() {
        let actual_tensor = heisenberg(1.2, -0.7, 0.3, false, false, Device::Cpu);
        let expected_tensor = raw_heisenberg_reference(1.2, -0.7, 0.3, false, false);
        assert!(actual_tensor.allclose(&expected_tensor, 1e-5, 1e-8, false));

        let actual_matrix = heisenberg(1.2, -0.7, 0.3, false, true, Device::Cpu);
        let expected_matrix = raw_heisenberg_reference(1.2, -0.7, 0.3, false, true);
        assert!(actual_matrix.allclose(&expected_matrix, 1e-5, 1e-8, false));
    }
}
