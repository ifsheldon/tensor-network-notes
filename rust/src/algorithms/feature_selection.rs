//! Entanglement-based feature-selection helpers.

use tch::{IndexOp, Kind, Tensor};

use crate::feature_mapping::{cossin_feature_map, feature_map_to_qubit_state};
use crate::mps::MPS;
use crate::quantum_state::{calc_onsite_entanglement_entropy, project_state};
use crate::types::OrthogonalizationMode;

/// Dynamic onsite entanglement entropy analysis values.
pub fn dyn_oee_analyze(samples: &Tensor, nth_img: i64) -> Tensor {
    assert_eq!(samples.dim(), 2, "samples must be a 2D tensor");
    assert!(samples.size()[1] >= 3);
    let num_samples = samples.size()[0];
    let features = cossin_feature_map(samples, 0.5, true);
    let states = feature_map_to_qubit_state(&features);
    let tensor_state =
        states.sum_dim_intlist([0].as_slice(), false, None::<Kind>) / (num_samples as f64).sqrt();
    let total_oee = calc_onsite_entanglement_entropy(&tensor_state, None, 1e-14).sum(None::<Kind>);
    let mut changes = Vec::new();
    for feature_idx in 0..samples.size()[1] {
        let projected = project_state(
            &tensor_state,
            &features.i((nth_img, feature_idx, ..)),
            feature_idx,
        );
        let new_oee = calc_onsite_entanglement_entropy(&projected, None, 1e-14);
        changes.push(new_oee.sum(None::<Kind>) - &total_oee);
    }
    Tensor::stack(&changes, 0)
}

/// OEE variation after one-qubit measurements.
pub fn oee_variation_one_qubit_measurement(
    mps: &mut MPS,
    feature: &Tensor,
    oee_threshold: Option<f64>,
) -> Tensor {
    assert_eq!(feature.dim(), 2, "features must be a 2D tensor");
    let feature = feature.to_device(mps.device()).to_kind(mps.kind());
    let feature_num = feature.size()[0] as usize;
    let mut oees = mps.onsite_entanglement_entropy(None, 1e-10);
    let selected: Vec<usize> = if let Some(threshold) = oee_threshold {
        assert!(threshold > 0.0, "oee_eps must be a positive number");
        let mask = oees.ge(threshold);
        oees = oees.where_self(&mask, &Tensor::zeros_like(&oees));
        mask.nonzero()
            .squeeze_dim(-1)
            .iter::<i64>()
            .expect("indices")
            .map(|idx| idx as usize)
            .collect()
    } else {
        (0..feature_num).collect()
    };
    let oee_sum = oees.sum(None::<Kind>);
    let zero = Tensor::zeros_like(&oee_sum);
    let mut changes = Vec::new();
    for idx in 0..feature_num {
        if selected.contains(&idx) {
            mps.center_orthogonalize(idx as isize, OrthogonalizationMode::Qr, None, true, false);
            let mut projected = mps.project_one_qubit_to_state(idx, &feature.i(idx as i64));
            projected.set_center(Some(idx.saturating_sub(1).min(projected.len() - 1)));
            let new_indices: Vec<usize> = selected
                .iter()
                .copied()
                .filter(|&pos| pos != idx)
                .map(|pos| if pos > idx { pos - 1 } else { pos })
                .collect();
            let new_oees = if oee_threshold.is_some() {
                projected.onsite_entanglement_entropy(Some(&new_indices), 1e-10)
            } else {
                projected.onsite_entanglement_entropy(None, 1e-10)
            };
            changes.push(&oee_sum - new_oees.sum(None::<Kind>));
        } else {
            changes.push(zero.shallow_clone());
        }
    }
    Tensor::stack(&changes, 0)
}

/// Select feature indices by repeatedly measuring the largest onsite entropy.
pub fn entanglement_ordered_sampling_protocol(
    mps: &MPS,
    select_feature_num: Option<usize>,
) -> Tensor {
    let select_feature_num = select_feature_num.unwrap_or_else(|| mps.len());
    assert!(0 < select_feature_num && select_feature_num <= mps.len());
    let mut selected = Vec::new();
    let mut feature_indices: Vec<i64> = (0..mps.len() as i64).collect();
    let mut current = mps.shallow_clone();
    for _ in 0..select_feature_num {
        if feature_indices.len() == 1 {
            selected.push(feature_indices[0]);
            break;
        }
        current.center_orthogonalize(0, OrthogonalizationMode::Qr, None, true, true);
        let oees = current.onsite_entanglement_entropy(None, 1e-10);
        let argmax = oees.argmax(None, false).int64_value(&[]) as usize;
        selected.push(feature_indices[argmax]);
        let rdm = current.one_body_reduced_density_matrix(argmax, true, true);
        let (eigvals, eigvecs) = Tensor::internal_linalg_eigh(&rdm, "L", true);
        let project_to_state = eigvecs.i((.., eigvals.argmax(None, false).int64_value(&[])));
        current = current.project_one_qubit_to_state(argmax, &project_to_state);
        feature_indices.remove(argmax);
    }
    Tensor::from_slice(&selected)
        .to_kind(Kind::Int64)
        .to_device(mps.device())
}
