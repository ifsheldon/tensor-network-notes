use crate::utils::*;
use tch::{Kind, Tensor};

/// Inverse a permutation.
pub fn inverse_permutation(permutation: &[UIdx]) -> Vec<UIdx> {
    let perm: Vec<Idx> = permutation.iter().copied().cast_items().collect();
    let permutation = Tensor::from_slice(&perm);
    let mut inv = Tensor::empty_like(&permutation);
    let arange = Tensor::arange(
        permutation.size1().unwrap(),
        (permutation.kind(), permutation.device()),
    );
    let v: Vec<Idx> = inv.scatter_(0, &permutation, &arange).try_into().unwrap();
    v.into_iter().cast_items().collect()
}

/// Unify the dtypes of two tensors to the most appropriate type.
pub fn unify_tensor_dtypes(t1: &Tensor, t2: &Tensor) -> (Tensor, Tensor) {
    let k1 = t1.kind();
    assert!(
        matches!(
            k1,
            Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
        ),
        "quantum_state must be a float or complex tensor, got {k1:?}"
    );
    let k2 = t2.kind();
    assert!(
        matches!(
            k2,
            Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
        ),
        "quantum_state must be a float or complex tensor, got {k2:?}"
    );
    if k1 == k2 {
        return (t1.shallow_clone(), t2.shallow_clone());
    }
    let convert = [
        (Kind::Float, Kind::ComplexFloat, Kind::ComplexFloat),
        (Kind::Double, Kind::ComplexFloat, Kind::ComplexDouble),
        (Kind::Float, Kind::ComplexDouble, Kind::ComplexDouble),
        (Kind::Double, Kind::ComplexDouble, Kind::ComplexDouble),
    ];
    for (d1, d2, td) in convert {
        if (k1 == d1 && k2 == d2) || (k1 == d2 && k2 == d1) {
            return (t1.to_kind(td), t2.to_kind(td));
        }
    }
    let raise = [
        (Kind::Float, Kind::Double),
        (Kind::ComplexFloat, Kind::ComplexDouble),
    ];
    for (d1, d2) in raise {
        if (k1 == d1 && k2 == d2) || (k1 == d2 && k2 == d1) {
            return (t1.to_kind(d2), t2.to_kind(d2));
        }
    }
    unreachable!("Unreachable in unify_tensor_dtypes");
}

/// Map a float kind to a complex kind.
pub fn map_float_kind_to_complex(kind: Kind) -> Kind {
    match kind {
        Kind::Float => Kind::ComplexFloat,
        Kind::Double => Kind::ComplexDouble,
        _ => panic!("dtype must be float32 or float64"),
    }
}

/// Map a float tensor to a complex tensor.
pub fn map_float_tensor_to_complex(t: &Tensor) -> Tensor {
    let to_kind = map_float_kind_to_complex(t.kind());
    t.to_kind(to_kind)
}

/// Convert a tensor representing a quantum gate into a matrix form.
/// The tensor should have an even number of dimensions, each of size 2.
///
/// # Arguments
/// * `t`: The tensor representing the quantum gate.
/// * `num_qubit`: The number of qubits the gate is acting on. If None, it is inferred from the tensor shape.
/// # Returns
/// The matrix form of the quantum gate tensor.
pub fn view_gate_tensor_as_matrix(t: &Tensor, num_qubit: Option<Num>) -> Tensor {
    let ndim = t.dim();
    assert!(
        ndim % 2 == 0,
        "Tensor must have an even number of dimensions"
    );
    assert!(
        t.size().iter().all(|&d| d == 2),
        "Tensor dimensions must be 2"
    );
    let qubit_count = check_quantum_gate(t, num_qubit, true).expect("invalid gate tensor");
    let d = 1_i64 << qubit_count; // 2^q
    t.view([d, d])
}

/// Convert a matrix representing a quantum gate into a tensor form.
/// The matrix should have dimensions (2^n, 2^n) for some n.
///
/// # Arguments
/// * `t`: The matrix representing the quantum gate.
/// * `num_qubit`: The number of qubits the gate is acting on. If None, it is inferred from the matrix shape.
/// # Returns
/// The tensor form of the quantum gate matrix.
pub fn view_gate_matrix_as_tensor(t: &Tensor, num_qubit: Option<Num>) -> Tensor {
    assert!(t.dim() == 2, "Matrix must have 2 dimensions");
    let sz = t.size();
    assert!(sz[0] == sz[1], "Matrix must be square");
    let qubit_count = check_quantum_gate(t, num_qubit.cast(), false).expect("invalid gate matrix");
    let dims = vec![2_i64; (qubit_count * 2) as usize];
    t.view(&dims[..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn test_inverse_perm() {
        let p = vec![2, 0, 1];
        assert_eq!(inverse_permutation(&p), vec![1, 2, 0]);
    }

    #[test]
    fn test_view_gate_tensor_matrix_roundtrip() {
        // 2-qubit identity
        let i = Tensor::eye(4, (Kind::Float, Device::Cpu));
        let t = view_gate_matrix_as_tensor(&i, Some(2));
        assert_eq!(t.size(), vec![2, 2, 2, 2]);
        let m = view_gate_tensor_as_matrix(&t, Some(2));
        assert_eq!(m.size(), vec![4, 4]);
    }
}
