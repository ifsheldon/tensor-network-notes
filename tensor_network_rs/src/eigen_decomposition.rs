use tch::{Kind, Tensor};
use crate::types::*;

/// Generate a random real symmetric matrix of the given dimension.
///
/// Mirrors the Python `rand_real_symmetric_matrix(dim)` utility used by
/// the power-iteration example.
pub fn rand_real_symmetric_matrix(dim: Num) -> Tensor {
    let dim: TInt = dim.cast();
    let m = Tensor::randn([dim, dim], (Kind::Float, tch::Device::Cpu));
    (&m + m.tr()) / 2.0
}

/// Power-iteration-like eigen solver via matrix exponential, matching the
/// Python `eigs_power(mat, which, v0)` behavior.
///
/// - `which` is one of "la", "sa", "lm", "sm" for largest/smallest
///   algebraic or magnitude.
/// - Returns `(eigenvalue, eigenvector)` with a unit-norm eigenvector.
pub fn eigs_power(mat: &Tensor, which: &str, v0: Option<&Tensor>) -> tch::Result<(Tensor, Tensor)> {
    let which = which.to_lowercase();
    let h = mat;
    assert!(h.size().len() == 2, "matrix must be 2D");
    // symmetry check (approximate)
    let ht = h.tr();
    let diff = (h - &ht).abs().sum(Kind::Float).double_value(&[]);
    assert!(diff < 1e-6, "matrix must be symmetric for eigs_power");

    let tau = 0.01_f64;
    let rho = match which.as_str() {
        // match Python: use exp(tau * H) and exp(tau * H^2) for LA/LM
        "la" => (h * tau).matrix_exp(),
        "lm" => ((h.matmul(h)) * tau).matrix_exp(),
        // and exp(-tau * H), exp(-tau * H^2) for SA/SM
        "sa" => (h * (-tau)).matrix_exp(),
        "sm" => ((h.matmul(h)) * (-tau)).matrix_exp(),
        _ => panic!("which must be one of la/lm/sa/sm"),
    };

    let iter_num = 8000;
    let tolerance = 1e-12;
    let mut v = if let Some(v0t) = v0 {
        v0t.shallow_clone()
    } else {
        let vv = Tensor::randn([h.size()[1]], (h.kind(), h.device()));
        &vv / vv.norm()
    };

    for _ in 0..iter_num {
        let v_next = rho.matmul(&v);
        let v_next_n = &v_next / v_next.norm();
        let diff = (v_next_n.copy() - v.copy()).norm().double_value(&[]);
        v = v_next_n;
        if diff < tolerance {
            break;
        }
    }

    // Use Rayleigh quotient for a more accurate eigenvalue estimate
    let v_n = &v / v.norm();
    let hv = h.matmul(&v_n);
    let eigenvalue = v_n.dot(&hv);
    let eigenvector = v_n;
    Ok((eigenvalue, eigenvector))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eigs_power_shapes_and_residual() {
        let h = rand_real_symmetric_matrix(5);
        let (eval, evec) = eigs_power(&h, "la", None).unwrap();
        assert_eq!(evec.size(), vec![5]);
        // Basic sanity: vector has unit-ish norm and finite eigenvalue
        let nrm = evec.norm().double_value(&[]);
        assert!(nrm > 0.9 && nrm < 1.1);
        let ev = eval.double_value(&[]);
        assert!(ev.is_finite());
    }

    fn check_residual(h: &Tensor, which: &str) {
        let (eval, evec) = eigs_power(h, which, None).unwrap();
        let r = h.matmul(&evec) - &evec * eval;
        let res = r.norm().double_value(&[]);
        assert!(res < 1e-5, "residual too large for {}: {}", which, res);
    }

    #[test]
    fn test_eigs_power_residual_all_modes() {
        let h = rand_real_symmetric_matrix(6);
        for m in ["la", "sa", "lm", "sm"] {
            check_residual(&h, m);
        }
    }
}
