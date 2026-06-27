//! Hybrid classical/ADQC classifier.

use tch::{Tensor, nn};

use crate::feature_mapping::{cossin_feature_map, feature_map_to_qubit_state};
use crate::networks::adqc::{ADQCNet, probabilities_adqc_classifier};
use crate::types::GatePattern;

/// A hybrid classifier that combines a fully connected layer with an ADQC network.
#[derive(Debug)]
pub struct FCADQCHybridClassifier {
    adqc_net: ADQCNet,
    fc1: nn::Linear,
    fc2: nn::Linear,
    feature_num: i64,
    class_num: i64,
}

impl FCADQCHybridClassifier {
    /// Construct the hybrid classifier.
    pub fn new(
        vs: &nn::Path<'_>,
        feature_num: i64,
        dim_mid: i64,
        class_num: i64,
        num_qubits: i64,
        num_adqc_layers: i64,
        adqc_gate_pattern: GatePattern,
        identity_init: bool,
        double_precision: bool,
    ) -> Self {
        let fc1 = nn::linear(vs.sub("fc1"), feature_num, dim_mid, Default::default());
        let fc2 = nn::linear(vs.sub("fc2"), dim_mid, num_qubits, Default::default());
        let adqc_net = ADQCNet::new(
            &vs.sub("adqc"),
            num_qubits,
            num_adqc_layers,
            adqc_gate_pattern,
            identity_init,
            double_precision,
        );
        Self {
            adqc_net,
            fc1,
            fc2,
            feature_num,
            class_num,
        }
    }

    /// Forward pass.
    pub fn forward(&self, data: &Tensor) -> Tensor {
        assert!(
            data.dim() == 2 && data.size()[1] == self.feature_num,
            "The input data must be a 2D tensor with the expected feature count"
        );
        let x = data.apply(&self.fc1).relu().apply(&self.fc2).tanh();
        let x = cossin_feature_map(&x, 0.5, false);
        let state = feature_map_to_qubit_state(&x);
        let state = self.adqc_net.forward(&state);
        probabilities_adqc_classifier(&state, self.class_num, true)
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    use super::*;

    #[test]
    fn hybrid_classifier_returns_probabilities() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model =
            FCADQCHybridClassifier::new(&vs.root(), 4, 3, 2, 2, 1, GatePattern::Brick, true, false);
        let data = Tensor::rand([5, 4], (Kind::Float, Device::Cpu));
        let probs = model.forward(&data);
        assert_eq!(probs.size(), vec![5, 2]);
        let sums = probs.sum_dim_intlist([1].as_slice(), false, None::<Kind>);
        assert!(sums.allclose(
            &Tensor::ones([5], (Kind::Float, Device::Cpu)),
            1e-4,
            1e-6,
            false
        ));
    }
}
