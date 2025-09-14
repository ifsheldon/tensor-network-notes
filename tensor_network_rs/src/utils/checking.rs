use tch::{Kind, Tensor};

/// Return true if two slices share any element in common.
pub fn iterable_have_common(a: &[i64], b: &[i64]) -> bool {
    use std::collections::HashSet;
    let set_a: HashSet<i64> = a.iter().copied().collect();
    b.iter().any(|x| set_a.contains(x))
}

/// Validate a tensor as a quantum state tensor: ndim>0 and every dim equals 2
/// and dtype is float or complex.
pub fn check_state_tensor(t: &Tensor) -> Result<(), String> {
    let kind = t.kind();
    match kind {
        Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble => {}
        _ => return Err("quantum_state must be a float or complex tensor".into()),
    }
    let size = t.size();
    if size.is_empty() {
        return Err("quantum_state must be a tensor with at least one dimension".into());
    }
    if !size.iter().all(|&d| d == 2) {
        return Err("quantum_state must be a tensor with all dimensions of size 2".into());
    }
    Ok(())
}

fn is_power_of_two(x: i64) -> bool {
    x > 0 && (x & (x - 1)) == 0
}

fn ilog2(x: i64) -> i64 {
    // assumes x is a power of two
    (63 - x.leading_zeros() as i64) - (63 - (x - 1).leading_zeros() as i64 - 1)
}

/// Validate a quantum gate tensor. Returns the inferred/validated number of qubits.
pub fn check_quantum_gate(
    t: &Tensor,
    num_qubits: Option<i64>,
    assert_tensor_form: bool,
) -> Result<i64, String> {
    let kind = t.kind();
    match kind {
        Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble => {}
        _ => return Err("quantum_gate must be a float or complex tensor".into()),
    }
    let ndim = t.dim();
    if ndim < 2 {
        return Err("quantum_gate must be a tensor with at least two dimensions".into());
    }
    if ndim % 2 != 0 {
        return Err("quantum_gate must have an even number of dimensions".into());
    }

    let sizes = t.size();
    if ndim == 2 {
        let m = sizes[0];
        let n = sizes[1];
        let nq = if let Some(nq) = num_qubits {
            nq
        } else {
            // infer from dimension
            if !is_power_of_two(m) {
                return Err(format!("matrix dim {} is not 2^k", m));
            }
            ilog2(m)
        };
        if m != n || m != 2_i64.pow(nq as u32) {
            return Err(format!(
                "gate must be {}x{} square matrix",
                2_i64.pow(nq as u32),
                2_i64.pow(nq as u32)
            ));
        }
        if assert_tensor_form && nq > 1 {
            return Err("Quantum gate should be in tensor form".into());
        }
        Ok(nq)
    } else {
        if !sizes.iter().all(|&d| d == 2) {
            return Err("gate tensor must have all dimensions of size 2".into());
        }
        let nq = num_qubits.unwrap_or((ndim / 2) as i64);
        if ndim as i64 != 2 * nq {
            return Err(format!(
                "gate tensor must have 2 * num_qubits dimensions, got {}",
                ndim
            ));
        }
        Ok(nq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind, Tensor};

    #[test]
    fn test_state_and_gate_checks() {
        let s = Tensor::zeros([2, 2, 2], (Kind::Float, Device::Cpu));
        assert!(check_state_tensor(&s).is_ok());
        let i4 = Tensor::eye(4, (Kind::Float, Device::Cpu));
        assert_eq!(check_quantum_gate(&i4, None, false).unwrap(), 2);
        let g = Tensor::zeros([2, 2, 2, 2], (Kind::Float, Device::Cpu));
        assert_eq!(check_quantum_gate(&g, None, true).unwrap(), 2);
    }
}
