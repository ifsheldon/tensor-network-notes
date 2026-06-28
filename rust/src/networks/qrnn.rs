//! ADQC recurrent network helpers.

use einops::einops;
use tch::{Device, Kind, Tensor, nn};

use crate::feature_mapping::{cossin_feature_map, feature_map_to_qubit_state};
use crate::networks::adqc::ADQCNet;
use crate::types::GatePattern;
use crate::utils::tensors::zeros_state;

/// ADQC-based quantum recurrent network.
#[derive(Debug)]
pub struct ADQCRNN {
    num_aux_qubits: i64,
    num_feature_qubits: i64,
    net: ADQCNet,
    complex_kind: Kind,
}

impl ADQCRNN {
    /// Construct an ADQC RNN.
    pub fn new(
        vs: &nn::Path<'_>,
        num_aux_qubits: i64,
        num_feature_qubits: i64,
        num_layers: i64,
        gate_pattern: GatePattern,
        identity_init: bool,
        double_precision: bool,
    ) -> Self {
        assert!(num_aux_qubits > 0);
        assert!(num_feature_qubits > 0);
        assert!(num_layers > 0);
        let net = ADQCNet::new(
            &vs.sub("adqc_rnn"),
            num_aux_qubits + num_feature_qubits,
            num_layers,
            gate_pattern,
            identity_init,
            double_precision,
        );
        let complex_kind = if double_precision {
            Kind::ComplexDouble
        } else {
            Kind::ComplexFloat
        };
        Self {
            num_aux_qubits,
            num_feature_qubits,
            net,
            complex_kind,
        }
    }

    /// Forward pass for data shaped `(batch, sample_length, feature_dim)`.
    pub fn forward(&self, data_batch: &Tensor) -> Tensor {
        assert_eq!(data_batch.dim(), 3, "data_batch must be a 3D tensor");
        let (batch, sample_length, feature_dim) = data_batch.size3().expect("3D");
        assert_eq!(feature_dim, self.num_feature_qubits);
        let device = data_batch.device();
        let aux_flat_dim = 2_i64.pow(self.num_aux_qubits as u32);
        let feature_flat_dim = 2_i64.pow(self.num_feature_qubits as u32);
        let batch_usize = batch as usize;
        let aux_state =
            zeros_state(self.num_aux_qubits, self.complex_kind, Device::Cpu).to_device(device);
        let mut aux = einops!(".. -> {batch_usize} (..)", &aux_state);
        let mut norms = Tensor::ones([batch, 1], (data_batch.kind(), device));
        for t in 0..sample_length {
            let features = cossin_feature_map(&data_batch.select(1, t), 1.0, true);
            let feature_state =
                feature_map_to_qubit_state(&features).reshape([batch, feature_flat_dim]);
            let state = aux.unsqueeze(2) * feature_state.unsqueeze(1);
            let mut state_shape = vec![batch];
            state_shape.extend(vec![
                2;
                (self.num_aux_qubits + self.num_feature_qubits) as usize
            ]);
            let state = self.net.forward(&state.reshape(state_shape.as_slice()));
            let state = state.reshape([batch, aux_flat_dim, feature_flat_dim]);
            let projected = state.select(2, 0);
            norms = projected.norm_scalaropt_dim(2.0, [1].as_slice(), true);
            aux = &projected / &norms;
        }
        norms.squeeze_dim(1)
    }
}

/// Calculate a series of combined sine and cosine waves.
pub fn series_sin_cos(length: i64, coeff_sin: &Tensor, coeff_cos: &Tensor, k_step: f64) -> Tensor {
    assert_eq!(coeff_sin.dim(), 1);
    assert_eq!(coeff_cos.dim(), 1);
    assert_eq!(coeff_sin.numel(), coeff_cos.numel());
    assert_eq!(coeff_sin.device(), coeff_cos.device());
    let order = coeff_sin.numel() as i64;
    let time_steps = Tensor::arange(length, (coeff_sin.kind(), coeff_sin.device())).unsqueeze(0);
    let orders = Tensor::arange(order, (coeff_sin.kind(), coeff_sin.device())).unsqueeze(1);
    let y_sin = (time_steps.shallow_clone() * (&orders * k_step)).sin() * coeff_sin.unsqueeze(1);
    let y_cos = (time_steps * (&orders * k_step)).cos() * coeff_cos.unsqueeze(1);
    y_sin.sum_dim_intlist([0].as_slice(), false, None::<Kind>)
        + y_cos.sum_dim_intlist([0].as_slice(), false, None::<Kind>)
}

/// Prepare sliding-window samples from a one-dimensional series.
pub fn prepare_series_samples(series: &Tensor, sample_length: i64, step_size: i64) -> Tensor {
    assert!(sample_length >= step_size && step_size >= 1);
    let length = series.size()[0];
    assert!(length >= sample_length);
    let mut samples = Vec::new();
    let mut start = 0;
    while start < length - sample_length {
        samples.push(series.narrow(0, start, sample_length));
        start += step_size;
    }
    Tensor::stack(&samples, 0)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor, nn};

    use super::*;

    #[test]
    fn adqc_rnn_forward_returns_one_norm_per_sample() {
        let vs = nn::VarStore::new(Device::Cpu);
        let model = ADQCRNN::new(&vs.root(), 1, 1, 1, GatePattern::Brick, true, false);
        let data = Tensor::rand([3, 4, 1], (Kind::Float, Device::Cpu));
        let out = model.forward(&data);
        assert_eq!(out.size(), vec![3]);
    }

    #[test]
    fn prepare_series_samples_uses_sliding_windows() {
        let series = Tensor::arange(6, (Kind::Float, Device::Cpu));
        let samples = prepare_series_samples(&series, 3, 2);
        assert_eq!(samples.size(), vec![2, 3]);
        let expected = Tensor::from_slice(&[0.0_f32, 1.0, 2.0, 2.0, 3.0, 4.0]).reshape([2, 3]);
        assert!(samples.allclose(&expected, 1e-5, 1e-8, false));
    }
}
