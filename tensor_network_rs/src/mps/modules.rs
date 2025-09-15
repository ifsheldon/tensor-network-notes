use crate::mps::functional::{
    MPSType, calc_global_tensor_by_tensordot, calc_inner_product, calculate_mps_norm_factors,
    gen_random_mps_tensors, orthogonalize_arange, project_multi_qubits_vec, tt_decomposition,
};
use tch::{Device, Kind, Tensor};

#[allow(dead_code)]
pub struct MPS {
    mps: Vec<Tensor>,
    length: usize,
    physical_dim: i64,
    virtual_dim: i64,
    mps_type: MPSType,
    dtype: Kind,
    device: Device,
    requires_grad: bool,
    center: Option<usize>,
}

impl MPS {
    pub fn from_tensors(mps_tensors: Vec<Tensor>, requires_grad: Option<bool>) -> Self {
        assert!(!mps_tensors.is_empty());
        let length = mps_tensors.len();
        let physical_dim = mps_tensors[0].size()[1];
        let virtual_dim = mps_tensors[0].size()[2];
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

    pub fn random(
        length: i64,
        physical_dim: i64,
        virtual_dim: i64,
        mps_type: MPSType,
        dtype: Kind,
        device: Device,
        requires_grad: bool,
    ) -> Self {
        let mps =
            gen_random_mps_tensors(length, physical_dim, virtual_dim, mps_type, dtype, device);
        Self {
            mps,
            length: length as usize,
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
    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    pub fn center(&self) -> Option<usize> {
        self.center
    }
    pub fn local_tensors(&self) -> &Vec<Tensor> {
        &self.mps
    }

    pub fn center_orthogonalization(
        &mut self,
        mut center: isize,
        mode: &str,
        truncate_dim: Option<i64>,
        check_nan: bool,
        normalize: bool,
    ) {
        assert!(center >= -(self.length as isize) && center < self.length as isize);
        if center < 0 {
            center += self.length as isize;
        }
        let center_usize = center as usize;
        if self.center.is_none() {
            let (mps2, _) = orthogonalize_arange(
                &self.mps,
                0,
                center_usize,
                mode,
                truncate_dim,
                normalize,
                false,
                check_nan,
            );
            let (mps3, _) = orthogonalize_arange(
                &mps2,
                self.length - 1,
                center_usize,
                mode,
                truncate_dim,
                normalize,
                false,
                check_nan,
            );
            self.mps = mps3;
        } else if self.center.unwrap() != center_usize {
            let (mps2, changed) = orthogonalize_arange(
                &self.mps,
                self.center.unwrap(),
                center_usize,
                mode,
                truncate_dim,
                normalize,
                true,
                check_nan,
            );
            if let Some(ixs) = changed {
                for i in ixs {
                    self.mps[i] = mps2[i].shallow_clone();
                }
            }
        }
        self.center = Some(center_usize);
        if normalize {
            self.center_normalize();
        }
    }

    pub fn center_normalize(&mut self) {
        let c = self.center.expect("not center-orthogonalized");
        let n = self.mps[c].norm();
        self.mps[c] = &self.mps[c] / n;
    }

    pub fn force_set_local_tensor(&mut self, i: usize, value: Tensor) {
        let v = value.to_kind(self.dtype).to_device(self.device);
        self.mps[i] = v;
    }

    pub fn global_tensor(&self) -> Tensor {
        if self.length > 15 { /* warning omitted in Rust */ }
        calc_global_tensor_by_tensordot(&self.mps)
    }

    pub fn norm_factors(&self) -> Tensor {
        calculate_mps_norm_factors(&self.mps, true).real()
    }

    pub fn norm(&self, efficient: bool) -> Tensor {
        if efficient && self.center.is_some() {
            return self.mps[self.center.unwrap()].norm();
        }
        let f = self.norm_factors();
        f.sqrt().prod(f.kind())
    }

    pub fn normalize_(&mut self, efficient: bool) {
        if efficient && self.center.is_some() {
            let c = self.center.unwrap();
            self.mps[c] = &self.mps[c] / self.norm(true);
            return;
        }
        let f = 1.0f64 / self.norm_factors().sqrt();
        for i in 0..self.length {
            let s = f.double_value(&[i as i64]);
            self.mps[i] = &self.mps[i] * s;
        }
    }

    pub fn inner_product(&self, other: &MPS, return_product_factors: bool) -> Tensor {
        assert_eq!(self.length, other.length);
        let factors = calc_inner_product(&self.mps, &other.mps);
        if return_product_factors {
            factors
        } else {
            factors.prod(factors.kind())
        }
    }

    pub fn project_multi_qubits_vec(&self, qubit_indices: &[i64], states: &Tensor) -> MPS {
        let new_locals = project_multi_qubits_vec(&self.mps, qubit_indices, states);
        MPS::from_tensors(new_locals, Some(self.requires_grad))
    }

    pub fn project_multi_qubits_indices(&self, qubit_indices: &[i64], states_idx: &[i64]) -> MPS {
        let new_locals = crate::mps::functional::project_multi_qubits_indices(
            &self.mps,
            qubit_indices,
            states_idx,
        );
        MPS::from_tensors(new_locals, Some(self.requires_grad))
    }

    pub fn from_state_tensor(state_tensor: &Tensor, max_rank: Option<i64>, use_svd: bool) -> MPS {
        let (locals, _clipped) = tt_decomposition(state_tensor, max_rank, use_svd);
        let mut m = MPS::from_tensors(locals, Some(false));
        m.center = Some(m.length - 1);
        m
    }

    // RDM utilities
    pub fn one_body_reduced_density_matrix(
        &mut self,
        idx: usize,
        _do_tracing: bool,
        inplace_mutation: bool,
    ) -> Tensor {
        assert!(idx < self.length);
        if self.center.is_none() {
            if inplace_mutation {
                self.center_orthogonalization(idx as isize, "qr", None, true, true);
            } else {
                // out-of-place: shallow-clone tensors
                let tmp_vec: Vec<Tensor> = self.mps.iter().map(|t| t.shallow_clone()).collect();
                let mut tmp = MPS::from_tensors(tmp_vec, Some(self.requires_grad));
                tmp.center_orthogonalization(idx as isize, "qr", None, true, true);
                return tmp.one_body_reduced_density_matrix(idx, _do_tracing, true);
            }
        } else if self.center.unwrap() != idx {
            self.center_orthogonalization(idx as isize, "qr", None, true, true);
        }
        let t = &self.mps[idx]; // [left, physical, right]
        // Contract over left/right: l p r, l p' r -> p p'
        Tensor::einsum(
            "l p r, l q r -> p q",
            &[t.conj(), t.shallow_clone()],
            None::<Vec<i64>>,
        )
    }

    pub fn two_body_reduced_density_matrix(
        &mut self,
        idx0: usize,
        idx1: usize,
        return_matrix: bool,
    ) -> Tensor {
        assert!(idx0 < idx1 && idx1 < self.length);
        // Simpler and reliable: build the global state and reduce
        let state = self.global_tensor(); // [2,2,...,2]
        let keep = vec![idx0 as i64, idx1 as i64];
        let rdm = crate::quantum_state::functional::calc_reduced_density_matrix(&state, keep);
        if return_matrix {
            rdm
        } else {
            rdm.reshape([2, 2, 2, 2])
        }
    }

    pub fn entanglement_entropy_onsite_(
        &mut self,
        indices: Option<Vec<usize>>,
        eps: f64,
    ) -> Tensor {
        let idxs: Vec<usize> = match indices {
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
            let l = Tensor::f_from_slice(&[l1, l2]).unwrap();
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
        let n = m.norm(true).double_value(&[]);
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
