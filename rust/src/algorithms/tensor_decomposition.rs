//! Tensor-decomposition helpers.

use tch::{Kind, Tensor, no_grad};

use crate::utils::tensors::outer_product;

/// Stopping criterion for iterative rank-1 decomposition.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Rank1StopCriterion {
    /// Stop when the scalar overlap converges.
    Zeta,
    /// Stop when vector and norm updates converge.
    Norms,
}

fn random_unit_vectors_like(tensor: &Tensor) -> Vec<Tensor> {
    tensor
        .size()
        .into_iter()
        .map(|dim| {
            let vector = Tensor::randn([dim], (tensor.kind(), tensor.device()));
            &vector / vector.norm()
        })
        .collect()
}

fn real_kind_for(kind: Kind) -> Kind {
    match kind {
        Kind::Double | Kind::ComplexDouble => Kind::Double,
        _ => Kind::Float,
    }
}

fn normalized_initial_vectors(tensor: &Tensor, initial_vectors: Option<&[Tensor]>) -> Vec<Tensor> {
    assert!(tensor.dim() >= 2, "tensor order must be at least 2");
    match initial_vectors {
        None => random_unit_vectors_like(tensor),
        Some(vectors) => {
            assert_eq!(
                vectors.len(),
                tensor.dim(),
                "initial vector count must match tensor order"
            );
            vectors
                .iter()
                .enumerate()
                .map(|(idx, vector)| {
                    assert_eq!(vector.dim(), 1, "initial vectors must be 1D");
                    assert_eq!(
                        vector.size()[0],
                        tensor.size()[idx],
                        "initial vector length mismatch at index {idx}"
                    );
                    assert_eq!(
                        vector.device(),
                        tensor.device(),
                        "initial vectors must be on the same device as tensor"
                    );
                    assert_eq!(
                        vector.kind(),
                        tensor.kind(),
                        "initial vectors must have the same dtype as tensor"
                    );
                    vector / vector.norm()
                })
                .collect()
        }
    }
}

fn contract_rank1_update(tensor: &Tensor, vectors: &[Tensor], idx: usize) -> Tensor {
    let mut contracted = tensor.shallow_clone();
    for vector in vectors.iter().take(idx) {
        contracted = contracted.tensordot(&vector.conj(), [0], [0]);
    }
    for vector in vectors.iter().skip(idx + 1).rev() {
        contracted = contracted.tensordot(&vector.conj(), [-1], [0]);
    }
    contracted
}

fn rank1_vector_update(tensor: &Tensor, vectors: &[Tensor], idx: usize) -> Tensor {
    let vectors_without_idx = vectors
        .iter()
        .enumerate()
        .filter_map(|(j, vector)| (j != idx).then(|| vector.conj()))
        .collect::<Vec<_>>();
    let outer = outer_product(&vectors_without_idx).unsqueeze(idx as i64);
    let sum_indices = (0..tensor.dim() as i64)
        .filter(|&dim| dim != idx as i64)
        .collect::<Vec<_>>();
    (tensor * outer).sum_dim_intlist(sum_indices.as_slice(), false, None::<Kind>)
}

fn rank1_overlap(tensor: &Tensor, vectors: &[Tensor]) -> Tensor {
    let conjugated = vectors.iter().map(Tensor::conj).collect::<Vec<_>>();
    (tensor * outer_product(&conjugated))
        .sum(None::<Kind>)
        .real()
}

/// Rank-1 tensor completion algorithm adapted from the reference implementation.
pub fn rank1_tc(
    tensor: &Tensor,
    initial_vectors: Option<&[Tensor]>,
    num_iter: i64,
    eps: f64,
) -> (Vec<Tensor>, Tensor) {
    assert!(num_iter > 0, "num_iter must be positive");
    assert!(eps > 0.0, "eps must be positive");
    let mut vectors = normalized_initial_vectors(tensor, initial_vectors);
    let mut norm_prev: Option<Tensor> = None;
    for _ in 0..num_iter {
        let mut vector_diff = 0.0;
        let mut norm_diff = 0.0;
        for idx in 0..tensor.dim() {
            let update = contract_rank1_update(tensor, &vectors, idx);
            let norm = update.norm();
            let vector = &update / &norm;
            vector_diff += (&vectors[idx] - &vector).norm().double_value(&[]);
            norm_diff += norm_prev
                .as_ref()
                .map(|prev| (&norm - prev).abs().double_value(&[]))
                .unwrap_or(f64::INFINITY);
            vectors[idx] = vector;
            norm_prev = Some(norm);
        }
        let order = tensor.dim() as f64;
        if vector_diff / order < eps && norm_diff / order < eps {
            break;
        }
    }
    (vectors, norm_prev.expect("num_iter is positive"))
}

/// Decompose a tensor into a dominant rank-1 component by alternating updates.
pub fn rank1_decomposition(
    tensor: &Tensor,
    num_iter: i64,
    stop_criterion: Rank1StopCriterion,
    eps: f64,
) -> (Vec<Tensor>, Tensor) {
    assert!(tensor.dim() >= 2, "tensor order must be at least 2");
    assert!(num_iter > 0, "num_iter must be positive");
    assert!(eps > 0.0, "eps must be positive");
    let mut vectors = random_unit_vectors_like(tensor);
    let mut zeta = Tensor::ones([], (real_kind_for(tensor.kind()), tensor.device()));
    match stop_criterion {
        Rank1StopCriterion::Zeta => {
            for _ in 0..num_iter {
                for idx in 0..tensor.dim() {
                    let update = rank1_vector_update(tensor, &vectors, idx);
                    vectors[idx] = &update / update.norm();
                }
                let zeta_new = rank1_overlap(tensor, &vectors);
                if (&zeta_new - &zeta).abs().double_value(&[]) < eps {
                    zeta = zeta_new;
                    break;
                }
                zeta = zeta_new;
            }
        }
        Rank1StopCriterion::Norms => {
            for _ in 0..num_iter {
                let mut vector_diff = 0.0;
                let mut zeta_diff = 0.0;
                for idx in 0..tensor.dim() {
                    let update = rank1_vector_update(tensor, &vectors, idx);
                    let update_norm = update.norm();
                    let vector = &update / &update_norm;
                    vector_diff += (&vectors[idx] - &vector).norm().double_value(&[]);
                    zeta_diff += (&update_norm - &zeta).abs().double_value(&[]);
                    vectors[idx] = vector;
                    zeta = update_norm;
                }
                let order = tensor.dim() as f64;
                if vector_diff / order < eps && zeta_diff / order < eps {
                    break;
                }
            }
        }
    }
    (vectors, zeta)
}

/// Decompose a tensor into a rank-1 component using autograd on temporary vectors.
pub fn rank1_decomposition_gradient_based(
    tensor: &Tensor,
    num_iter: i64,
    eps: f64,
    learning_rate: f64,
) -> (Vec<Tensor>, Tensor) {
    assert!(tensor.dim() >= 2, "tensor order must be at least 2");
    assert!(num_iter > 0, "num_iter must be positive");
    assert!(eps > 0.0, "eps must be positive");
    assert!(learning_rate > 0.0, "learning_rate must be positive");
    let mut vectors = random_unit_vectors_like(tensor)
        .into_iter()
        .map(|vector| vector.set_requires_grad(true))
        .collect::<Vec<_>>();
    let mut zeta = Tensor::ones([], (real_kind_for(tensor.kind()), tensor.device()));
    for _ in 0..num_iter {
        let outer = outer_product(&vectors);
        let loss = (tensor - outer).norm();
        loss.backward();
        no_grad(|| {
            for vector in &mut vectors {
                let grad = vector.grad();
                *vector -= grad * learning_rate;
            }
        });
        for vector in &mut vectors {
            vector.zero_grad();
        }
        let norms = vectors.iter().map(Tensor::norm).collect::<Vec<_>>();
        let zeta_new = Tensor::stack(&norms, 0).prod(None::<Kind>);
        if (&zeta_new - &zeta).abs().double_value(&[]) < eps {
            zeta = zeta_new;
            break;
        }
        zeta = zeta_new;
    }
    let vectors = no_grad(|| {
        vectors
            .into_iter()
            .map(|vector| {
                let normalized = &vector / vector.norm();
                normalized.detach()
            })
            .collect::<Vec<_>>()
    });
    (vectors, zeta)
}

/// Convert a tensor into a matrix by moving `left_index` to the front.
pub fn make_matrix(tensor: &Tensor, left_index: i64) -> Tensor {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    assert!(0 <= left_index && left_index < order);
    tensor
        .movedim_int(left_index, 0)
        .reshape([tensor.size()[left_index as usize], -1])
}

/// Tucker decomposition using SVD on each mode.
pub fn tucker_decomposition(tensor: &Tensor) -> (Tensor, Vec<Tensor>, Vec<i64>) {
    let order = tensor.dim() as i64;
    assert!(order >= 2);
    let mut matrices_u = Vec::new();
    let mut ranks = Vec::new();
    for idx in 0..order {
        let matrix = make_matrix(tensor, idx);
        let (u, s, _) = Tensor::linalg_svd(&matrix, false, "");
        let rank = s
            .ne(0.0)
            .to_kind(Kind::Int64)
            .sum(None::<Kind>)
            .int64_value(&[]);
        matrices_u.push(u);
        ranks.push(rank);
    }
    let mut core = tensor.shallow_clone();
    for matrix in &matrices_u {
        core = core.tensordot(&matrix.conj(), [0], [0]);
    }
    (core, matrices_u, ranks)
}

/// Reduced matrix of a Tucker core.
pub fn reduced_matrix(core_tensor: &Tensor, n: i64) -> Tensor {
    let order = core_tensor.dim() as i64;
    assert!(order >= 2);
    assert!(0 <= n && n < order);
    let matrix = make_matrix(core_tensor, n);
    matrix.matmul(&matrix.conj().transpose(0, 1))
}

/// Rank-1 tensor from vectors.
pub fn rank1_tensor(vectors: &[Tensor]) -> Tensor {
    outer_product(vectors)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::*;

    #[test]
    fn rank1_tc_recovers_exact_rank1_tensor_from_good_initialization() {
        let v0 = Tensor::from_slice(&[0.6_f32, 0.8]);
        let v1 = Tensor::from_slice(&[1.0_f32, 0.0, 0.0]);
        let tensor = outer_product(&[v0.shallow_clone(), v1.shallow_clone()]) * 3.0;
        let (vectors, norm) = rank1_tc(&tensor, Some(&[v0, v1]), 10, 1e-6);
        let reconstructed = outer_product(&vectors) * norm;
        assert!(reconstructed.allclose(&tensor, 1e-5, 1e-6, false));
    }

    #[test]
    fn rank1_decomposition_returns_unit_vectors() {
        tch::manual_seed(0);
        let tensor = Tensor::randn([2, 3, 2], (Kind::Float, Device::Cpu));
        let (vectors, _zeta) = rank1_decomposition(&tensor, 5, Rank1StopCriterion::Zeta, 1e-12);
        assert_eq!(vectors.len(), 3);
        for vector in vectors {
            assert!((vector.norm().double_value(&[]) - 1.0).abs() < 1e-5);
        }
    }
}
