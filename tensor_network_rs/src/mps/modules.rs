use crate::mps::functional::{
    MPSType, calc_global_tensor_by_tensordot, calc_inner_product, calculate_mps_norm_factors,
    gen_random_mps_tensors, orthogonalize_arange, project_multi_qubits_vec, tt_decomposition,
};
use crate::utils::*;
use tch::{Device, Kind, Tensor};
#[allow(dead_code)]
/// Matrix Product State (MPS) container mirroring the Python class.
///
/// Provides utilities for center orthogonalization, normalization, norms,
/// projections, reduced density matrices, and basic conversions.
pub struct MPS {
    mps: Vec<Tensor>,
    length: Num,
    physical_dim: Num,
    virtual_dim: Num,
    mps_type: MPSType,
    dtype: Kind,
    device: Device,
    requires_grad: bool,
    center: Option<Num>,
}

impl MPS {
    /// Construct an MPS from a vector of local tensors.
    pub fn from_tensors(mps_tensors: Vec<Tensor>, requires_grad: Option<bool>) -> Self {
        assert!(!mps_tensors.is_empty());
        let length = mps_tensors.len();
        let physical_dim = mps_tensors[0].size()[1].cast();
        let virtual_dim = mps_tensors[0].size()[2].cast();
        let mps_type = if mps_tensors[0].size()[0] == 1 {
            MPSType::Open
        } else {
            MPSType::Periodic
        };
        let dtype = mps_tensors[0].kind();
        let device = mps_tensors[0].device();
        let requires_grad = requires_grad.unwrap_or(false);
        Self {
            mps: mps_tensors,
            length: length.cast(),
            physical_dim,
            virtual_dim,
            mps_type,
            dtype,
            device,
            requires_grad,
            center: None,
        }
    }

    pub fn random(
        length: Num,
        physical_dim: Num,
        virtual_dim: Num,
        mps_type: MPSType,
        dtype: Kind,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let mps =
            gen_random_mps_tensors(length, physical_dim, virtual_dim, mps_type, dtype, device);
        Self {
            mps,
            length,
            physical_dim,
            virtual_dim,
            mps_type,
            dtype,
            device,
            requires_grad,
            center: None,
        }
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> Num {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn center(&self) -> Option<Num> {
        self.center
    }
    pub fn local_tensors(&self) -> &Vec<Tensor> {
        &self.mps
    }

    /// Perform center orthogonalization (in-place). Matches Python semantics
    /// for modes `"svd"`/`"qr"`, optional truncation, and optional normalization.
    pub fn center_orthogonalization(
        &mut self,
        mut center: Idx,
        mode: &str,
        truncate_dim: Option<Num>,
        check_nan: bool,
        normalize: bool,
    ) {
        let length: Idx = self.length.cast();
        assert!(center >= -length && center < length);
        if center < 0 {
            center += length;
        }
        let center: UIdx = center.cast();
        if self.center.is_none() {
            let (mps2, _) = orthogonalize_arange(
                &self.mps,
                0,
                center,
                mode,
                truncate_dim,
                normalize,
                false,
                check_nan,
            );
            let (mps3, _) = orthogonalize_arange(
                &mps2,
                self.length - 1,
                center,
                mode,
                truncate_dim,
                normalize,
                false,
                check_nan,
            );
            self.mps = mps3;
        } else if self.center.unwrap() != center {
            let (mps2, changed) = orthogonalize_arange(
                &self.mps,
                self.center.unwrap(),
                center,
                mode,
                truncate_dim,
                normalize,
                true,
                check_nan,
            );
            if let Some(ixs) = changed {
                for i in ixs {
                    let i: usize = i.cast();
                    self.mps[i] = mps2[i].shallow_clone();
                }
            }
        }
        self.center = Some(center);
        if normalize {
            self.center_normalize();
        }
    }

    /// Normalize the center tensor in-place to unit norm.
    pub fn center_normalize(&mut self) {
        let c = self.center.expect("not center-orthogonalized");
        let c: usize = c.cast();
        let n = self.mps[c].norm();
        self.mps[c] = &self.mps[c] / n;
    }

    /// Force set a local tensor with dtype/device checks, mirroring Python guardrails.
    pub fn force_set_local_tensor(&mut self, i: usize, value: Tensor) {
        let v = value.to_kind(self.dtype).to_device(self.device);
        self.mps[i] = v;
    }

    /// Calculate the global state tensor by contracting all local tensors.
    pub fn global_tensor(&self) -> Tensor {
        if self.length > 15 { /* warning omitted in Rust */ }
        calc_global_tensor_by_tensordot(&self.mps)
    }

    /// Per-site norm factors (real), following the Python helper.
    pub fn norm_factors(&self) -> Tensor {
        calculate_mps_norm_factors(&self.mps).real()
    }

    /// Norm of the MPS; when `efficient=true` and center is set, uses the
    /// center tensor norm shortcut, otherwise multiplies sqrt of factors.
    pub fn norm(&self) -> Tensor {
        if enable_efficient_mode() && self.center.is_some() {
            let c: usize = self.center.unwrap().cast();
            return self.mps[c].norm();
        }
        let f = self.norm_factors();
        f.sqrt().prod(f.kind())
    }

    /// Normalize the MPS in-place (either via center tensor or per-site factors).
    pub fn normalize_(&mut self) {
        if enable_efficient_mode() && self.center.is_some() {
            let c: usize = self.center.unwrap().cast();
            self.mps[c] = &self.mps[c] / self.norm();
            return;
        }
        let f = 1.0f64 / self.norm_factors().sqrt();
        for i in 0..self.length {
            let i: usize = i.cast();
            let s = f.double_value(&[i.to_tint()]);
            self.mps[i] = &self.mps[i] * s;
        }
    }

    /// Inner product with another MPS; optionally return the per-site product factors.
    pub fn inner_product(&self, other: &MPS, return_product_factors: bool) -> Tensor {
        assert_eq!(self.length, other.length);
        let factors = calc_inner_product(&self.mps, &other.mps);
        if return_product_factors {
            factors
        } else {
            factors.prod(factors.kind())
        }
    }

    /// Project multiple qubits using projection vectors, returning a new MPS.
    pub fn project_multi_qubits_vec(&self, qubit_indices: &[UIdx], states: &Tensor) -> MPS {
        let new_locals = project_multi_qubits_vec(&self.mps, qubit_indices, states);
        MPS::from_tensors(new_locals, Some(self.requires_grad))
    }

    /// Project multiple qubits by choosing basis-state indices, returning a new MPS.
    pub fn project_multi_qubits_indices(&self, qubit_indices: &[UIdx], states_idx: &[UIdx]) -> MPS {
        let new_locals = crate::mps::functional::project_multi_qubits_indices(
            &self.mps,
            qubit_indices,
            states_idx,
        );
        MPS::from_tensors(new_locals, Some(self.requires_grad))
    }

    /// Initialize an MPS from a full state tensor via TT decomposition.
    pub fn from_state_tensor(state_tensor: &Tensor, max_rank: Option<Num>, use_svd: bool) -> MPS {
        let (locals, _clipped) = tt_decomposition(state_tensor, max_rank, use_svd);
        let mut m = MPS::from_tensors(locals, Some(false));
        m.center = Some(m.length - 1);
        m
    }

    // RDM utilities
    /// One-body reduced density matrix at `idx`. If needed, moves center.
    pub fn one_body_reduced_density_matrix(
        &mut self,
        idx: UIdx,
        do_tracing: bool,
        inplace_mutation: bool,
    ) -> Tensor {
        // TODO: split this into two functions `one_body_reduced_density_matrix` and `one_body_reduced_density_matrix_` (inplace)
        assert!(idx < self.length);
        if self.center.is_none() {
            if inplace_mutation {
                self.center_orthogonalization(idx.cast(), "qr", None, true, true);
            } else {
                // out-of-place: shallow-clone tensors
                let cloned_local_tensors: Vec<Tensor> = self
                    .mps
                    .iter()
                    .map(|tensor| tensor.shallow_clone())
                    .collect();
                let mut cloned_mps =
                    MPS::from_tensors(cloned_local_tensors, Some(self.requires_grad));
                cloned_mps.center_orthogonalization(idx.cast(), "qr", None, true, true);
                return cloned_mps.one_body_reduced_density_matrix(idx, do_tracing, true);
            }
        } else if self.center.unwrap() != idx {
            self.center_orthogonalization(idx.cast(), "qr", None, true, true);
        }
        let idx: usize = idx.cast();
        let center_tensor = &self.mps[idx]; // [left, physical, right]
        // Contract over left/right: l p r, l p' r -> p p'
        let mut reduced_density = Tensor::einsum(
            "l p r, l q r -> p q",
            &[center_tensor.conj(), center_tensor.shallow_clone()],
            NO_OPT_PATH,
        );
        if do_tracing {
            let trace = reduced_density.trace();
            reduced_density = &reduced_density / trace;
        }
        reduced_density
    }

    /// Two-body reduced density matrix on sites `(idx0, idx1)`; if `return_matrix`
    /// is true, returns `[4,4]`, otherwise `[2,2,2,2]`.
    pub fn two_body_reduced_density_matrix(
        &mut self,
        idx0: UIdx,
        idx1: UIdx,
        return_matrix: bool,
    ) -> Tensor {
        assert!(idx0 < idx1 && idx1 < self.length);
        // Simpler and reliable: build the global state and reduce
        let state = self.global_tensor(); // [2,2,...,2]
        let keep = vec![idx0, idx1];
        let rdm = crate::quantum_state::functional::calc_reduced_density_matrix(&state, keep);
        if return_matrix {
            rdm
        } else {
            rdm.reshape([2, 2, 2, 2])
        }
    }

    pub fn entanglement_entropy_onsite_(&mut self, indices: Option<Vec<UIdx>>, eps: f64) -> Tensor {
        let idxs: Vec<UIdx> = match indices {
            None => (0..self.length).collect(),
            Some(v) => v,
        };
        assert!(!idxs.is_empty() && idxs.iter().all(|&i| i < self.length));
        let mut ents: Vec<Tensor> = Vec::with_capacity(idxs.len());
        for &i in &idxs {
            let rdm = self.one_body_reduced_density_matrix(i, true, true);
            // Analytic eigenvalues of 2x2 Hermitian
            let a = rdm.double_value(&[0, 0]);
            let b = rdm.double_value(&[1, 1]);
            let c_re = rdm.real().double_value(&[0, 1]);
            let c_im = rdm.imag().double_value(&[0, 1]);
            let c_abs2 = c_re * c_re + c_im * c_im;
            let disc = ((a - b) * (a - b) + 4.0 * c_abs2).sqrt();
            let l1 = (a + b + disc) * 0.5;
            let l2 = (a + b - disc) * 0.5;
            let l = Tensor::from_slice(&[l1, l2]);
            let e = -(l.copy() * (l.copy() + eps).log()).sum(l.kind());
            ents.push(e);
        }
        Tensor::stack(&ents, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mps_global_norm() {
        let mut m = MPS::random(4, 2, 3, MPSType::Open, Kind::Float, Device::Cpu, false);
        m.center_orthogonalization(2, "qr", None, true, true);
        let n = m.norm().double_value(&[]);
        assert!(n.is_finite() && n > 0.0);
    }

    #[test]
    fn test_two_body_rdm_properties() {
        let mut m = MPS::random(5, 2, 4, MPSType::Open, Kind::Float, Device::Cpu, false);
        m.center_orthogonalization(2, "qr", None, true, true);
        let rdm = m.two_body_reduced_density_matrix(1, 2, true); // [4,4]
        // Hermitian: rdm == rdm^H
        let diff = (&rdm - rdm.conj().transpose(0, 1))
            .abs()
            .sum(rdm.kind())
            .double_value(&[]);
        assert!(diff < 1e-8);
        // Trace equals 1 (normalized)
        let tr = rdm.trace().real().double_value(&[]);
        assert!((tr - 1.0).abs() < 1e-6);
    }
}
