use crate::utils::checking::check_quantum_gate;
use tch::{Kind, Tensor};

pub fn inverse_permutation(p: &[i64]) -> Vec<i64> {
    let n = p.len();
    let mut inv = vec![0_i64; n];
    for (i, &pi) in p.iter().enumerate() {
        inv[pi as usize] = i as i64;
    }
    inv
}

pub fn unify_tensor_dtypes(t1: &Tensor, t2: &Tensor) -> (Tensor, Tensor) {
    let k1 = t1.kind();
    let k2 = t2.kind();
    let valid = matches!(
        k1,
        Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
    ) && matches!(
        k2,
        Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble
    );
    assert!(valid, "quantum_state must be a float or complex tensor");
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
    panic!("Unreachable in unify_tensor_dtypes");
}

pub fn map_float_kind_to_complex(kind: Kind) -> Kind {
    match kind {
        Kind::Float => Kind::ComplexFloat,
        Kind::Double => Kind::ComplexDouble,
        _ => panic!("dtype must be float32 or float64"),
    }
}

pub fn map_float_tensor_to_complex(t: &Tensor) -> Tensor {
    let to_kind = map_float_kind_to_complex(t.kind());
    t.to_kind(to_kind)
}

pub fn view_gate_tensor_as_matrix(t: &Tensor, num_qubit: Option<i64>) -> Tensor {
    let ndim = t.dim();
    assert!(
        ndim % 2 == 0,
        "Tensor must have an even number of dimensions"
    );
    assert!(
        t.size().iter().all(|&d| d == 2),
        "Tensor dimensions must be 2"
    );
    let qubit_count = num_qubit.unwrap_or((ndim / 2) as i64);
    check_quantum_gate(t, Some(qubit_count), false).expect("invalid gate tensor");
    let d = 1_i64 << qubit_count; // 2^q
    t.view([d, d])
}

pub fn view_gate_matrix_as_tensor(t: &Tensor, num_qubit: Option<i64>) -> Tensor {
    assert!(t.dim() == 2, "Matrix must have 2 dimensions");
    let sz = t.size();
    assert!(sz[0] == sz[1], "Matrix must be square");
    let inferred = ((sz[0] as f64).log2().round()) as i64;
    let qubit_count = num_qubit.unwrap_or(inferred);
    assert!(
        sz[0] == (1_i64 << qubit_count),
        "Matrix size must be (2^q, 2^q)"
    );
    check_quantum_gate(t, Some(qubit_count), false).expect("invalid gate matrix");
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
        let m = view_gate_tensor_as_matrix(&t, Some(2));
        assert_eq!(m.size(), vec![4, 4]);
    }
}
