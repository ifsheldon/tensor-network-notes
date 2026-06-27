//! Semantic checks preserved from the Python implementation.

use std::collections::HashSet;

use tch::{Kind, Tensor};

/// Return whether two integer slices share any element.
pub fn iterable_have_common(a: &[i64], b: &[i64]) -> bool {
    let lhs: HashSet<i64> = a.iter().copied().collect();
    b.iter().any(|x| lhs.contains(x))
}

/// Check that a tensor is a valid quantum-state tensor.
pub fn check_state_tensor(tensor: &Tensor) {
    assert!(
        is_float_or_complex(tensor.kind()),
        "quantum_state must be a float or complex tensor"
    );
    assert!(
        tensor.size().iter().all(|&x| x == 2),
        "quantum_state must be a tensor with all dimensions of size 2"
    );
    assert!(
        tensor.dim() > 0,
        "quantum_state must be a tensor with at least one dimension"
    );
}

/// Check that a tensor is a valid quantum gate and return the number of qubits it acts on.
pub fn check_quantum_gate(
    tensor: &Tensor,
    num_qubits: Option<i64>,
    assert_tensor_form: bool,
) -> i64 {
    assert!(
        is_float_or_complex(tensor.kind()),
        "quantum_gate must be a float or complex tensor"
    );
    let dims = tensor.size();
    assert!(
        dims.len() >= 2,
        "quantum_gate must be a tensor with at least two dimensions"
    );
    assert!(
        dims.len() % 2 == 0,
        "quantum_gate must have an even number of dimensions"
    );
    if dims.len() == 2 {
        let inferred = num_qubits.unwrap_or_else(|| log2_exact(dims[0]));
        assert!(
            dims[0] == dims[1] && dims[0] == 2_i64.pow(inferred as u32),
            "gate must be a square matrix with dimensions 2^num_qubits, got {dims:?}"
        );
        assert!(
            !(assert_tensor_form && inferred > 1),
            "Quantum gate should be in tensor form"
        );
        inferred
    } else {
        assert!(
            dims.iter().all(|&d| d == 2),
            "gate tensor must have all dimensions of size 2"
        );
        let inferred = num_qubits.unwrap_or((dims.len() / 2) as i64);
        assert_eq!(
            dims.len() as i64,
            2 * inferred,
            "gate tensor must have 2 * num_qubits dimensions"
        );
        inferred
    }
}

/// Whether a `tch::Kind` can represent state or gate amplitudes.
pub fn is_float_or_complex(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
    )
}

/// Whether a `tch::Kind` is a real floating dtype.
pub fn is_float(kind: Kind) -> bool {
    matches!(kind, Kind::Float | Kind::Double)
}

/// Map a real floating dtype to its complex counterpart.
pub fn complex_kind_for_float(kind: Kind) -> Kind {
    match kind {
        Kind::Float => Kind::ComplexFloat,
        Kind::Double => Kind::ComplexDouble,
        _ => panic!("dtype must be float32 or float64"),
    }
}

fn log2_exact(value: i64) -> i64 {
    assert!(value > 0, "dimension must be positive");
    assert!(
        value > 0 && (value & (value - 1)) == 0,
        "dimension must be an exact power of two, got {value}"
    );
    value.trailing_zeros() as i64
}
