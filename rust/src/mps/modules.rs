//! Matrix-product-state type.

use std::path::Path;

use einops::{einops, einsumstr};
use tch::{Device, IndexOp, Kind, Tensor};

use crate::error::{Result, TensorNetworkError};
use crate::mps::functional::{
    ProjectToStates, calc_global_tensor_by_tensordot, calc_inner_product,
    calculate_mps_norm_factors, gen_random_mps_tensors, normalize_mps, orthogonalize_arange,
    project_multi_qubits, tt_decomposition,
};
use crate::types::{MPSType, OrthogonalizationMode};

/// Matrix Product State.
#[allow(clippy::upper_case_acronyms)]
pub struct MPS {
    tensors: Vec<Tensor>,
    center: Option<usize>,
}

impl std::fmt::Debug for MPS {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MPS")
            .field("length", &self.len())
            .field("physical_dim", &self.physical_dim())
            .field("virtual_dim", &self.virtual_dim())
            .field("mps_type", &self.mps_type())
            .field("center", &self.center)
            .finish()
    }
}

impl MPS {
    /// Build an MPS from local tensors.
    pub fn from_tensors(tensors: Vec<Tensor>) -> Self {
        validate_mps_tensors(&tensors);
        Self {
            tensors,
            center: None,
        }
    }

    /// Build a random MPS.
    pub fn random(
        length: i64,
        physical_dim: i64,
        virtual_dim: i64,
        mps_type: MPSType,
        kind: Kind,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let tensors =
            gen_random_mps_tensors(length, physical_dim, virtual_dim, mps_type, kind, device);
        for tensor in &tensors {
            let _ = tensor.set_requires_grad(requires_grad);
        }
        Self::from_tensors(tensors)
    }

    /// Build an MPS from a state tensor using TT decomposition.
    pub fn from_state_tensor(state_tensor: &Tensor, max_rank: Option<i64>, use_svd: bool) -> Self {
        let (local_tensors, _) = tt_decomposition(state_tensor, max_rank, use_svd);
        let center = local_tensors.len() - 1;
        let mut mps = Self::from_tensors(local_tensors);
        mps.center = Some(center);
        mps
    }

    /// Return an explicit shallow clone of all tensor handles.
    pub fn shallow_clone(&self) -> Self {
        Self {
            tensors: self.tensors.iter().map(Tensor::shallow_clone).collect(),
            center: self.center,
        }
    }

    /// Save this MPS to a Python-compatible safetensors file.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut named: Vec<(String, Tensor)> = self
            .tensors
            .iter()
            .enumerate()
            .map(|(idx, tensor)| (idx.to_string(), tensor.contiguous()))
            .collect();
        let center_value = self.center.map_or(-1_i64, |idx| idx as i64);
        named.push((
            "center".to_string(),
            Tensor::from(center_value)
                .to_device(self.device())
                .contiguous(),
        ));
        let borrowed: Vec<(&str, &Tensor)> = named
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect();
        Tensor::write_safetensors(&borrowed, path)?;
        Ok(())
    }

    /// Load an MPS from a Python-compatible safetensors file.
    pub fn load_safetensors<P: AsRef<Path>>(path: P, requires_grad: bool) -> Result<Self> {
        let mut entries = Tensor::read_safetensors(path)?;
        let center_pos = entries
            .iter()
            .position(|(key, _)| key == "center")
            .ok_or_else(|| TensorNetworkError::MissingTensorKey("center".to_string()))?;
        let (_, center_tensor) = entries.remove(center_pos);
        let center_value = center_tensor.to_device(Device::Cpu).int64_value(&[]);
        let mut numbered = Vec::<(usize, Tensor)>::with_capacity(entries.len());
        for (key, tensor) in entries {
            let idx = key
                .parse::<usize>()
                .map_err(|_| TensorNetworkError::InvalidTensorKey(key.clone()))?;
            numbered.push((idx, tensor));
        }
        numbered.sort_by_key(|(idx, _)| *idx);
        for (expected, (actual, _)) in numbered.iter().enumerate() {
            if expected != *actual {
                return Err(TensorNetworkError::MissingTensorKey(expected.to_string()));
            }
        }
        let tensors: Vec<Tensor> = numbered
            .into_iter()
            .map(|(_, tensor)| {
                let _ = tensor.set_requires_grad(requires_grad);
                tensor
            })
            .collect();
        let mut mps = Self::from_tensors(tensors);
        mps.center = if center_value == -1 {
            None
        } else {
            Some(center_value as usize)
        };
        Ok(mps)
    }

    /// Number of local tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether this MPS has no local tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Physical dimension.
    pub fn physical_dim(&self) -> i64 {
        self.tensors[0].size()[1]
    }

    /// Representative virtual dimension.
    pub fn virtual_dim(&self) -> i64 {
        self.tensors[0].size()[2]
    }

    /// Boundary-condition type.
    pub fn mps_type(&self) -> MPSType {
        MPSType::from_tensors(&self.tensors)
    }

    /// Orthogonality center, if known.
    pub fn center(&self) -> Option<usize> {
        self.center
    }

    /// Center tensor, if known.
    pub fn center_tensor(&self) -> Option<&Tensor> {
        self.center.map(|idx| &self.tensors[idx])
    }

    /// Device of the local tensors.
    pub fn device(&self) -> Device {
        self.tensors[0].device()
    }

    /// Dtype of the local tensors.
    pub fn kind(&self) -> Kind {
        self.tensors[0].kind()
    }

    /// Shallow-cloned local tensors.
    pub fn local_tensors(&self) -> Vec<Tensor> {
        self.tensors.iter().map(Tensor::shallow_clone).collect()
    }

    /// Borrow one local tensor.
    pub fn local_tensor(&self, idx: usize) -> &Tensor {
        &self.tensors[idx]
    }

    /// Set center metadata after an operation that preserves or intentionally moves the center.
    pub fn set_center(&mut self, center: Option<usize>) {
        if let Some(center) = center {
            assert!(center < self.len(), "center out of range");
        }
        self.center = center;
    }

    /// Replace one local tensor after semantic shape and dtype checks.
    pub fn set_local_tensor(&mut self, idx: usize, value: Tensor) {
        assert_eq!(
            value.size(),
            self.tensors[idx].size(),
            "value shape must match local tensor shape"
        );
        assert_eq!(
            value.kind(),
            self.tensors[idx].kind(),
            "value dtype must match local tensor dtype"
        );
        self.force_set_local_tensor(idx, value);
    }

    /// Replace one local tensor after checking shape, preserving the incoming dtype/device.
    pub fn replace_local_tensor(&mut self, idx: usize, value: Tensor) {
        assert_eq!(
            value.size(),
            self.tensors[idx].size(),
            "value shape must match local tensor shape"
        );
        self.tensors[idx] = value;
    }

    /// Replace one local tensor after converting dtype/device to the current MPS.
    pub fn force_set_local_tensor(&mut self, idx: usize, value: Tensor) {
        let value = value.to_kind(self.kind()).to_device(self.device());
        let _ = value.set_requires_grad(self.tensors[idx].requires_grad());
        self.tensors[idx] = value;
    }

    /// Calculate the full global state tensor.
    pub fn global_tensor(&self) -> Tensor {
        calc_global_tensor_by_tensordot(&self.tensors)
    }

    /// Calculate norm factors.
    pub fn norm_factors(&self) -> Tensor {
        calculate_mps_norm_factors(&self.tensors, true).real()
    }

    /// Calculate the MPS norm.
    pub fn norm(&self, efficient_mode: bool) -> Tensor {
        if efficient_mode && self.center.is_some() {
            self.tensors[self.center.expect("checked center")].norm()
        } else {
            self.norm_factors().sqrt().prod(None::<Kind>)
        }
    }

    /// Normalize this MPS in place.
    pub fn normalize(&mut self, efficient_mode: bool) {
        if efficient_mode && self.center.is_some() {
            let idx = self.center.expect("checked center");
            self.tensors[idx] = &self.tensors[idx] / self.norm(true);
        } else {
            let factors = self.norm_factors().sqrt().reciprocal();
            for idx in 0..self.len() {
                self.tensors[idx] = &self.tensors[idx] * factors.i(idx as i64);
            }
        }
    }

    /// Return a normalized MPS.
    pub fn normalized(&self, efficient_mode: bool) -> Self {
        if efficient_mode && self.center.is_some() {
            let mut result = self.shallow_clone();
            result.normalize(true);
            result
        } else {
            let mut result = Self::from_tensors(normalize_mps(&self.tensors));
            result.center = self.center;
            result
        }
    }

    /// Calculate the inner product with another MPS.
    pub fn inner_product(&self, other: &Self, return_product_factors: bool) -> Tensor {
        assert_eq!(
            self.len(),
            other.len(),
            "length of two MPS must be the same"
        );
        let factors = calc_inner_product(&self.tensors, &other.tensors);
        if return_product_factors {
            factors
        } else {
            factors.prod(None::<Kind>)
        }
    }

    /// Move the orthogonality center in place.
    pub fn center_orthogonalize(
        &mut self,
        center: isize,
        mode: OrthogonalizationMode,
        truncate_dim: Option<i64>,
        check_nan: bool,
        normalize: bool,
    ) {
        let center = normalize_index(center, self.len());
        if self.center.is_none() {
            let (left_pass, _) = orthogonalize_arange(
                &self.tensors,
                0,
                center,
                mode,
                truncate_dim,
                normalize,
                check_nan,
            );
            let (new_tensors, _) = orthogonalize_arange(
                &left_pass,
                self.len() - 1,
                center,
                mode,
                truncate_dim,
                normalize,
                check_nan,
            );
            self.tensors = new_tensors;
        } else if self.center != Some(center) {
            let (new_tensors, changed_indices) = orthogonalize_arange(
                &self.tensors,
                self.center.expect("checked center"),
                center,
                mode,
                truncate_dim,
                false,
                check_nan,
            );
            for idx in changed_indices {
                self.tensors[idx] = new_tensors[idx].shallow_clone();
            }
        }
        self.center = Some(center);
        if normalize {
            self.center_normalize();
        }
    }

    /// Return an orthogonalized MPS.
    pub fn center_orthogonalized(
        &self,
        center: isize,
        mode: OrthogonalizationMode,
        truncate_dim: Option<i64>,
        check_nan: bool,
        normalize: bool,
    ) -> Self {
        let mut result = self.shallow_clone();
        result.center_orthogonalize(center, mode, truncate_dim, check_nan, normalize);
        result
    }

    /// Normalize the center tensor.
    pub fn center_normalize(&mut self) {
        let center = self.center.expect(
            "The MPS is not center orthogonalized. Perform center orthogonalization first.",
        );
        self.tensors[center] = &self.tensors[center] / self.tensors[center].norm();
    }

    /// Return this MPS on a new device.
    pub fn to_device(&self, device: Device) -> Self {
        let mut result = Self::from_tensors(
            self.tensors
                .iter()
                .map(|tensor| tensor.to_device(device))
                .collect(),
        );
        result.center = self.center;
        result
    }

    /// Move this MPS to a device in place.
    pub fn set_device(&mut self, device: Device) {
        for tensor in &mut self.tensors {
            *tensor = tensor.to_device(device);
        }
    }

    /// Return this MPS with a new dtype.
    pub fn to_kind(&self, kind: Kind) -> Self {
        let mut result = Self::from_tensors(
            self.tensors
                .iter()
                .map(|tensor| tensor.to_kind(kind))
                .collect(),
        );
        result.center = self.center;
        result
    }

    /// Change dtype in place.
    pub fn set_kind(&mut self, kind: Kind) {
        for tensor in &mut self.tensors {
            *tensor = tensor.to_kind(kind);
        }
    }

    /// Set autograd tracking for all local tensors.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        for tensor in &self.tensors {
            let _ = tensor.set_requires_grad(requires_grad);
        }
    }

    /// Calculate a one-body reduced density matrix.
    pub fn one_body_reduced_density_matrix(
        &mut self,
        idx: usize,
        do_tracing: bool,
        inplace_mutation: bool,
    ) -> Tensor {
        assert!(idx < self.len(), "idx must be in [0, length - 1]");
        let center_tensor = if self.center == Some(idx) {
            self.tensors[idx].shallow_clone()
        } else if inplace_mutation {
            self.center_orthogonalize(idx as isize, OrthogonalizationMode::Qr, None, true, false);
            self.tensors[idx].shallow_clone()
        } else {
            let local_tensors = self.local_tensors();
            let new_tensors = if let Some(center) = self.center {
                orthogonalize_arange(
                    &local_tensors,
                    center,
                    idx,
                    OrthogonalizationMode::Qr,
                    None,
                    false,
                    true,
                )
                .0
            } else {
                let (left, _) = orthogonalize_arange(
                    &local_tensors,
                    0,
                    idx,
                    OrthogonalizationMode::Qr,
                    None,
                    false,
                    true,
                );
                orthogonalize_arange(
                    &left,
                    self.len() - 1,
                    idx,
                    OrthogonalizationMode::Qr,
                    None,
                    false,
                    true,
                )
                .0
            };
            new_tensors[idx].shallow_clone()
        };
        let rdm = Tensor::einsum(
            einsumstr!("left physical right, left physical_conj right -> physical physical_conj"),
            &[&center_tensor, &center_tensor.conj()],
            None::<i64>,
        );
        if do_tracing { &rdm / rdm.trace() } else { rdm }
    }

    /// Project multiple qubits, returning a new MPS.
    pub fn project_multi_qubits(
        &self,
        qubit_indices: &[usize],
        project_to_states: ProjectToStates<'_>,
    ) -> Self {
        let projected = match project_to_states {
            ProjectToStates::Vectors(states) => {
                let states = states.to_device(self.device()).to_kind(self.kind());
                project_multi_qubits(
                    &self.tensors,
                    qubit_indices,
                    ProjectToStates::Vectors(&states),
                )
            }
            ProjectToStates::Indices(indices) => project_multi_qubits(
                &self.tensors,
                qubit_indices,
                ProjectToStates::Indices(indices),
            ),
        };
        Self::from_tensors(projected)
    }

    /// Project one qubit, returning a new MPS.
    pub fn project_one_qubit_to_index(&self, qubit_idx: usize, project_to_state: i64) -> Self {
        self.project_multi_qubits(&[qubit_idx], ProjectToStates::Indices(&[project_to_state]))
    }

    /// Project one qubit to a vector state, returning a new MPS.
    pub fn project_one_qubit_to_state(&self, qubit_idx: usize, project_to_state: &Tensor) -> Self {
        assert_eq!(
            project_to_state.dim(),
            1,
            "project_to_state must be a 1D tensor"
        );
        let states = project_to_state.unsqueeze(0);
        self.project_multi_qubits(&[qubit_idx], ProjectToStates::Vectors(&states))
    }

    /// Calculate onsite entanglement entropies using in-place center moves.
    pub fn onsite_entanglement_entropy(&mut self, indices: Option<&[usize]>, eps: f64) -> Tensor {
        let owned_indices;
        let indices = match indices {
            Some(indices) => indices,
            None => {
                owned_indices = (0..self.len()).collect::<Vec<_>>();
                &owned_indices
            }
        };
        assert!(
            !indices.is_empty() && indices.len() <= self.len(),
            "indices must be a list of indices in [0, length)"
        );
        let rdms: Vec<Tensor> = indices
            .iter()
            .map(|&idx| self.one_body_reduced_density_matrix(idx, true, true))
            .collect();
        let rdms = Tensor::stack(&rdms, 0);
        let eigvals = rdms.linalg_eigvalsh("L");
        let probs = &eigvals / eigvals.sum_dim_intlist([1].as_slice(), true, None::<Kind>);
        -((&probs) * (&probs + eps).log()).sum_dim_intlist([1].as_slice(), false, None::<Kind>)
    }

    /// Calculate a two-body reduced density matrix.
    pub fn two_body_reduced_density_matrix(
        &mut self,
        qubit_idx0: usize,
        qubit_idx1: usize,
        return_matrix: bool,
    ) -> Tensor {
        assert!(qubit_idx0 < qubit_idx1);
        self.center_orthogonalize(
            qubit_idx0 as isize,
            OrthogonalizationMode::Qr,
            None,
            true,
            true,
        );
        let tensor_left = &self.tensors[qubit_idx0];
        let mut product = Tensor::einsum(
            einsumstr!(
                "left physical_conj right_conj, left physical right -> physical_conj physical right_conj right"
            ),
            &[&tensor_left.conj(), tensor_left],
            None::<i64>,
        );
        for idx in qubit_idx0 + 1..qubit_idx1 {
            let tensor_i = &self.tensors[idx];
            product = Tensor::einsum(
                einsumstr!(
                    "i0_physical_conj i0_physical left_conj left, left_conj physical right_conj, left physical right -> i0_physical_conj i0_physical right_conj right"
                ),
                &[&product, &tensor_i.conj(), tensor_i],
                None::<i64>,
            );
        }
        let tensor_right = &self.tensors[qubit_idx1];
        let rdm = Tensor::einsum(
            einsumstr!(
                "i0_physical_conj i0_physical left_conj left, left_conj i1_physical_conj right, left i1_physical right -> i0_physical i1_physical i0_physical_conj i1_physical_conj"
            ),
            &[&product, &tensor_right.conj(), tensor_right],
            None::<i64>,
        );
        if return_matrix {
            einops!("i0 i1 i0_conj i1_conj -> (i0 i1) (i0_conj i1_conj)", &rdm)
        } else {
            rdm
        }
    }
}

fn validate_mps_tensors(tensors: &[Tensor]) {
    assert!(!tensors.is_empty(), "MPS must have at least one tensor");
    let kind = tensors[0].kind();
    let device = tensors[0].device();
    let physical_dim = tensors[0].size()[1];
    for (idx, tensor) in tensors.iter().enumerate() {
        let shape = tensor.size();
        assert_eq!(
            shape.len(),
            3,
            "MPS local tensor at index {idx} must be rank-3"
        );
        assert_eq!(
            shape[1], physical_dim,
            "all MPS local tensors must share the same physical dimension"
        );
        assert_eq!(tensor.kind(), kind, "all MPS tensors must share dtype");
        assert_eq!(tensor.device(), device, "all MPS tensors must share device");
        if idx + 1 < tensors.len() {
            assert_eq!(
                shape[2],
                tensors[idx + 1].size()[0],
                "neighboring MPS virtual dimensions must match"
            );
        }
    }
}

fn normalize_index(index: isize, length: usize) -> usize {
    assert!(length > 0, "length must be positive");
    let len = length as isize;
    assert!(-len <= index && index < len, "center out of range");
    if index < 0 {
        (len + index) as usize
    } else {
        index as usize
    }
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn random_open_mps_has_expected_shapes() {
        let mps = MPS::random(4, 2, 3, MPSType::Open, Kind::Float, Device::Cpu, false);
        assert_eq!(mps.len(), 4);
        assert_eq!(mps.local_tensor(0).size(), vec![1, 2, 3]);
        assert_eq!(mps.local_tensor(1).size(), vec![3, 2, 3]);
        assert_eq!(mps.local_tensor(3).size(), vec![3, 2, 1]);
        assert_eq!(mps.mps_type(), MPSType::Open);
    }

    #[test]
    fn global_tensor_has_physical_shape() {
        let mps = MPS::random(3, 2, 2, MPSType::Open, Kind::Float, Device::Cpu, false);
        assert_eq!(mps.global_tensor().size(), vec![2, 2, 2]);
    }

    #[test]
    fn normalized_mps_has_unit_norm() {
        let mps = MPS::random(4, 2, 2, MPSType::Open, Kind::Float, Device::Cpu, false);
        let normalized = mps.normalized(false);
        let norm = normalized.norm(false).double_value(&[]);
        assert!((norm - 1.0).abs() < 1e-4, "norm={norm}");
    }

    #[test]
    fn safetensors_roundtrip_preserves_center() {
        let mut mps = MPS::random(3, 2, 2, MPSType::Open, Kind::Float, Device::Cpu, false);
        mps.center_orthogonalize(1, OrthogonalizationMode::Qr, None, true, false);
        let file = NamedTempFile::new().expect("temp file");
        mps.save_safetensors(file.path()).expect("save");
        let loaded = MPS::load_safetensors(file.path(), false).expect("load");
        assert_eq!(loaded.center(), Some(1));
        assert_eq!(loaded.local_tensor(0).size(), mps.local_tensor(0).size());
    }

    fn raw_two_body_rdm_python_order(mps: &mut MPS, i0: usize, i1: usize) -> Tensor {
        mps.center_orthogonalize(i0 as isize, OrthogonalizationMode::Qr, None, true, true);
        let tensor_left = mps.local_tensor(i0);
        let mut product = Tensor::einsum(
            "apr,aqs->pqrs",
            &[&tensor_left.conj(), tensor_left],
            None::<i64>,
        );
        for idx in i0 + 1..i1 {
            let tensor_i = mps.local_tensor(idx);
            product = Tensor::einsum(
                "ijab,apc,bpd->ijcd",
                &[&product, &tensor_i.conj(), tensor_i],
                None::<i64>,
            );
        }
        let tensor_right = mps.local_tensor(i1);
        Tensor::einsum(
            "ijab,apc,bqc->jqip",
            &[&product, &tensor_right.conj(), tensor_right],
            None::<i64>,
        )
    }

    #[test]
    fn two_body_rdm_matches_raw_python_order_reference() {
        let base = MPS::random(4, 2, 3, MPSType::Open, Kind::Float, Device::Cpu, false);
        let mut actual_mps = base.shallow_clone();
        let mut expected_mps = base.shallow_clone();
        let actual = actual_mps.two_body_reduced_density_matrix(1, 3, false);
        let expected = raw_two_body_rdm_python_order(&mut expected_mps, 1, 3);
        assert!(actual.allclose(&expected, 1e-5, 1e-6, false));

        let mut actual_mps = base.shallow_clone();
        let mut expected_mps = base.shallow_clone();
        let actual_matrix = actual_mps.two_body_reduced_density_matrix(1, 3, true);
        let expected_matrix =
            raw_two_body_rdm_python_order(&mut expected_mps, 1, 3).reshape([4, 4]);
        assert!(actual_matrix.allclose(&expected_matrix, 1e-5, 1e-6, false));
    }
}
