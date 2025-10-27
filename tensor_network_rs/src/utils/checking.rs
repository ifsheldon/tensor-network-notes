use crate::utils::*;
use std::collections::HashSet;
use tch::{Kind, Tensor};
use warrant::warrant;

/// Return true if two slices share any element in common.
pub fn iterable_have_common(a: &[i64], b: &[i64]) -> bool {
    let set_a: HashSet<i64> = a.iter().copied().collect();
    b.iter().any(|x| set_a.contains(x))
}

/// Validate a tensor as a quantum state tensor: ndim>0 and every dim equals 2
/// and dtype is float or complex.
pub fn check_state_tensor(t: &Tensor) -> Result<()> {
    warrant!(matches!(t.kind(), Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble), else {
        bail!("quantum_state must be a float or complex tensor");
    });
    let size = t.size();
    if size.is_empty() {
        bail!("quantum_state must be a tensor with at least one dimension");
    }
    warrant!(size.iter().all(|&d| d == 2), else {
        bail!("quantum_state must be a tensor with all dimensions of size 2");
    });
    Ok(())
}

/// Validate a quantum gate tensor. Returns the inferred/validated number of qubits.
pub fn check_quantum_gate(
    t: &Tensor,
    num_qubits: Option<Num>,
    assert_tensor_form: bool,
) -> Result<Num> {
    warrant!(matches!(t.kind(), Kind::Float | Kind::Double | Kind::ComplexFloat | Kind::ComplexDouble), else {
        bail!("quantum_gate must be a float or complex tensor");
    });
    let ndim: Num = t.dim().cast();
    warrant!(ndim >= 2, else {
        bail!("quantum_gate must be a tensor with at least two dimensions");
    });
    warrant!(ndim % 2 == 0, else {
        bail!("quantum_gate must have an even number of dimensions");
    });

    let sizes = t.size();
    if ndim == 2 {
        // in matrix form
        let m = sizes[0];
        let n = sizes[1];
        warrant!(m == n, else {
            bail!("gate must be a square matrix");
        });
        let nq = if let Some(nq) = num_qubits {
            warrant!(m == 2_i64.pow(nq as u32), else {
                bail!("gate must be a square matrix with dimensions 2^num_qubits, got {m}x{m}");
            });
            nq
        } else {
            let pow = m.ilog2();
            warrant!(2_i64.pow(pow) == m, else {
                bail!("matrix dim {} is not 2^k", m);
            });
            pow.cast()
        };
        if assert_tensor_form && nq > 1 {
            bail!("Quantum gate should be in tensor form");
        }
        Ok(nq)
    } else {
        // in tensor form
        warrant!(sizes.iter().all(|&d| d == 2), else {
            bail!("gate tensor must have all dimensions of size 2");
        });
        let nq = num_qubits.unwrap_or(ndim / 2);
        warrant!(ndim == 2 * nq, else {
            bail!(
                "gate tensor must have 2 * num_qubits dimensions, got {}",
                ndim
            );
        });
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
