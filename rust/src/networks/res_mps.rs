//! Residual MPS network.

use einops::einops;
use tch::{Kind, Tensor, nn};

/// A basic residual MPS model.
#[derive(Debug)]
pub struct ResMPSSimple {
    local_tensors: Vec<Tensor>,
    contract_vector_left: Tensor,
    contract_vector_right: Tensor,
    num_features: i64,
    num_classes: i64,
    feature_dim: i64,
    class_idx: i64,
}

impl ResMPSSimple {
    /// Construct a residual MPS model under an `nn::Path`.
    pub fn new(
        vs: &nn::Path<'_>,
        num_features: i64,
        feature_dim: i64,
        num_classes: i64,
        virtual_dim: i64,
        eps_norm: f64,
    ) -> Self {
        assert!(num_features > 0, "num_features must be positive");
        assert!(num_classes > 0, "num_classes must be positive");
        assert!(feature_dim > 0, "feature_dim must be positive");
        assert!(virtual_dim > 0, "virtual_dim must be positive");
        let class_idx = num_features / 2;
        let mut local_tensors = Vec::new();
        for idx in 0..num_features {
            let shape = if idx == class_idx {
                vec![virtual_dim, feature_dim, virtual_dim, num_classes]
            } else {
                vec![virtual_dim, feature_dim, virtual_dim]
            };
            let raw = Tensor::randn(shape.as_slice(), (Kind::Float, vs.device()));
            let tensor = (&raw / raw.norm()) * eps_norm;
            local_tensors.push(vs.add(&format!("local_{idx}"), tensor, true));
        }
        let contract = Tensor::ones([virtual_dim], (Kind::Float, vs.device()));
        let contract = &contract / contract.norm();
        let left = vs.add("contract_left", contract.shallow_clone(), false);
        let right = vs.add("contract_right", contract, false);
        Self {
            local_tensors,
            contract_vector_left: left,
            contract_vector_right: right,
            num_features,
            num_classes,
            feature_dim,
            class_idx,
        }
    }

    /// Forward pass for features shaped `(batch, num_features, feature_dim)`.
    pub fn forward(&self, features: &Tensor) -> Tensor {
        let (batch, num_features, feature_dim) = features.size3().expect("3D features");
        assert_eq!(num_features, self.num_features);
        assert_eq!(feature_dim, self.feature_dim);
        let batch_usize = batch as usize;
        let mut latent_left = einops!(
            "virtual_left -> {batch_usize} virtual_left",
            &self.contract_vector_left
        );
        for idx in 0..self.class_idx {
            let latent = Tensor::einsum(
                "lfr,bl,bf->br",
                &[
                    &self.local_tensors[idx as usize],
                    &latent_left,
                    &features.select(1, idx),
                ],
                None::<i64>,
            );
            latent_left = latent + latent_left;
        }
        let mut latent_right = einops!(
            "virtual_right -> {batch_usize} virtual_right",
            &self.contract_vector_right
        );
        for idx in ((self.class_idx + 1)..num_features).rev() {
            let latent = Tensor::einsum(
                "lfr,br,bf->bl",
                &[
                    &self.local_tensors[idx as usize],
                    &latent_right,
                    &features.select(1, idx),
                ],
                None::<i64>,
            );
            latent_right = latent + latent_right;
        }
        let activation = Tensor::einsum(
            "lfrc,bl,br,bf->bc",
            &[
                &self.local_tensors[self.class_idx as usize],
                &latent_left,
                &latent_right,
                &features.select(1, self.class_idx),
            ],
            None::<i64>,
        );
        assert_eq!(activation.size()[1], self.num_classes);
        activation
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    use super::*;

    #[test]
    fn residual_mps_forward_shape_matches_classes() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = ResMPSSimple::new(&vs.root(), 5, 2, 3, 4, 1e-4);
        let features = Tensor::rand([7, 5, 2], (Kind::Float, Device::Cpu));
        let out = model.forward(&features);
        assert_eq!(out.size(), vec![7, 3]);
    }
}
