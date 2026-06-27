//! Shape, dtype, and permutation helpers.

use tch::{Kind, Tensor};

use crate::utils::checking::{
    check_quantum_gate, complex_kind_for_float, is_float, is_float_or_complex,
};

/// Invert a permutation.
pub fn inverse_permutation(permutation: &[i64]) -> Vec<i64> {
    let mut inverse = vec![0; permutation.len()];
    for (idx, &value) in permutation.iter().enumerate() {
        assert!(
            0 <= value && (value as usize) < permutation.len(),
            "permutation value out of range"
        );
        inverse[value as usize] = idx as i64;
    }
    inverse
}

/// Convert tensors to a common dtype using the Python port's promotion rules.
pub fn unify_tensor_dtypes(t1: &Tensor, t2: &Tensor) -> (Tensor, Tensor) {
    let k1 = t1.kind();
    let k2 = t2.kind();
    assert!(
        is_float_or_complex(k1),
        "quantum_state must be a float or complex tensor"
    );
    assert!(
        is_float_or_complex(k2),
        "quantum_state must be a float or complex tensor"
    );
    if k1 == k2 {
        return (t1.shallow_clone(), t2.shallow_clone());
    }
    let target = match (k1, k2) {
        (Kind::Float, Kind::ComplexFloat) | (Kind::ComplexFloat, Kind::Float) => Kind::ComplexFloat,
        (Kind::Double, Kind::ComplexFloat) | (Kind::ComplexFloat, Kind::Double) => {
            Kind::ComplexDouble
        }
        (Kind::Float, Kind::ComplexDouble) | (Kind::ComplexDouble, Kind::Float) => {
            Kind::ComplexDouble
        }
        (Kind::Double, Kind::ComplexDouble) | (Kind::ComplexDouble, Kind::Double) => {
            Kind::ComplexDouble
        }
        (Kind::Float, Kind::Double) | (Kind::Double, Kind::Float) => Kind::Double,
        (Kind::ComplexFloat, Kind::ComplexDouble) | (Kind::ComplexDouble, Kind::ComplexFloat) => {
            Kind::ComplexDouble
        }
        _ => unreachable!("unreachable dtype promotion branch"),
    };
    (t1.to_kind(target), t2.to_kind(target))
}

/// Map a floating tensor to its complex counterpart.
pub fn map_float_tensor_to_complex(tensor: &Tensor) -> Tensor {
    assert!(is_float(tensor.kind()), "dtype must be float32 or float64");
    tensor.to_kind(complex_kind_for_float(tensor.kind()))
}

/// Map a floating dtype to its complex counterpart.
pub fn map_float_kind_to_complex(kind: Kind) -> Kind {
    complex_kind_for_float(kind)
}

/// Convert a gate tensor to matrix form.
pub fn view_gate_tensor_as_matrix(tensor: &Tensor, num_qubit: Option<i64>) -> Tensor {
    assert_eq!(
        tensor.dim() % 2,
        0,
        "Tensor must have an even number of dimensions"
    );
    assert!(
        tensor.size().iter().all(|&d| d == 2),
        "Tensor dimensions must be 2"
    );
    let qubit_count = num_qubit.unwrap_or((tensor.dim() / 2) as i64);
    check_quantum_gate(tensor, Some(qubit_count), false);
    tensor.view([2_i64.pow(qubit_count as u32), 2_i64.pow(qubit_count as u32)])
}

/// Convert a gate matrix to tensor form.
pub fn view_gate_matrix_as_tensor(tensor: &Tensor, num_qubit: Option<i64>) -> Tensor {
    let shape = tensor.size();
    assert_eq!(shape.len(), 2, "Matrix must have 2 dimensions");
    assert_eq!(shape[0], shape[1], "Matrix must be square");
    let qubit_count = num_qubit.unwrap_or_else(|| {
        assert!(
            shape[0] > 0 && (shape[0] & (shape[0] - 1)) == 0,
            "matrix dimension must be a power of two"
        );
        shape[0].trailing_zeros() as i64
    });
    assert_eq!(
        shape[0],
        2_i64.pow(qubit_count as u32),
        "Matrix size must be (2^q, 2^q) for q qubits"
    );
    check_quantum_gate(tensor, Some(qubit_count), false);
    let shape = vec![2; (qubit_count * 2) as usize];
    tensor.view(shape.as_slice())
}
