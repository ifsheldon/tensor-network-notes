//! Small eigen-decomposition helpers.

use tch::{Device, Kind, Tensor};

/// Eigenpair target for [`eigs_power`].
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EigenSelection {
    /// Largest algebraic eigenvalue.
    LargestAlgebraic,
    /// Smallest algebraic eigenvalue.
    SmallestAlgebraic,
    /// Largest magnitude eigenvalue.
    LargestMagnitude,
    /// Smallest magnitude eigenvalue.
    SmallestMagnitude,
}

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

/// Power-method eigenpair approximation used in the notebooks.
pub fn eigs_power(
    mat: &Tensor,
    which: EigenSelection,
    initial_vector: Option<&Tensor>,
    tau: f64,
    num_iter: i64,
    tolerance: f64,
) -> (Tensor, Tensor) {
    assert_eq!(mat.dim(), 2, "mat must be 2D");
    assert_eq!(mat.size()[0], mat.size()[1], "mat must be square");
    assert!(mat.allclose(&mat.transpose(0, 1), 1e-5, 1e-8, false));
    assert!(tau > 0.0, "tau must be positive");
    assert!(num_iter > 0, "num_iter must be positive");
    assert!(tolerance > 0.0, "tolerance must be positive");
    let rho = match which {
        EigenSelection::LargestAlgebraic => (mat * tau).matrix_exp(),
        EigenSelection::LargestMagnitude => ((mat.matrix_power(2)) * tau).matrix_exp(),
        EigenSelection::SmallestAlgebraic => (mat * -tau).matrix_exp(),
        EigenSelection::SmallestMagnitude => ((mat.matrix_power(2)) * -tau).matrix_exp(),
    };
    let mut v = match initial_vector {
        Some(vector) => {
            assert_eq!(vector.dim(), 1, "initial vector must be 1D");
            assert_eq!(
                vector.size()[0],
                mat.size()[1],
                "initial vector length mismatch"
            );
            assert_eq!(
                vector.device(),
                mat.device(),
                "initial vector must be on the same device as mat"
            );
            assert_eq!(
                vector.kind(),
                mat.kind(),
                "initial vector must have the same dtype as mat"
            );
            vector / vector.norm()
        }
        None => {
            let vector = Tensor::randn([mat.size()[1]], (mat.kind(), mat.device()));
            &vector / vector.norm()
        }
    };
    let mut norm = Tensor::ones([], (mat.kind(), mat.device()));
    for _ in 0..num_iter {
        let v_next = rho.matmul(&v);
        norm = v_next.norm();
        let v_next = &v_next / &norm;
        let diff = (&v_next - &v).norm();
        v = v_next;
        if diff.double_value(&[]) < tolerance {
            break;
        }
    }
    let scaled_eigenvector = mat.matmul(&v);
    let sign = v.dot(&scaled_eigenvector).sign();
    let eigenvector = &scaled_eigenvector / scaled_eigenvector.norm();
    let eigenvalue = match which {
        EigenSelection::LargestAlgebraic => norm.log() / tau,
        EigenSelection::SmallestAlgebraic => -norm.log() / tau,
        EigenSelection::LargestMagnitude => sign * (norm.log() / tau).sqrt(),
        EigenSelection::SmallestMagnitude => sign * (-norm.log() / tau).sqrt(),
    };
    (eigenvalue, eigenvector)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn eigs_power_finds_largest_and_smallest_algebraic_values() {
        let mat = Tensor::from_slice(&[2.0_f32, 0.0, 0.0, -1.0]).reshape([2, 2]);
        let initial = Tensor::from_slice(&[1.0_f32, 1.0]).to_device(Device::Cpu);
        let (largest, _) = eigs_power(
            &mat,
            EigenSelection::LargestAlgebraic,
            Some(&initial),
            0.1,
            200,
            1e-8,
        );
        let (smallest, _) = eigs_power(
            &mat,
            EigenSelection::SmallestAlgebraic,
            Some(&initial),
            0.1,
            200,
            1e-8,
        );
        assert!((largest.double_value(&[]) - 2.0).abs() < 1e-4);
        assert!((smallest.double_value(&[]) + 1.0).abs() < 1e-4);
        assert_eq!(largest.kind(), Kind::Float);
    }
}
