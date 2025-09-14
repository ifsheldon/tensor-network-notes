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

    pub fn from_state_tensor(state_tensor: &Tensor, max_rank: Option<i64>, use_svd: bool) -> MPS {
        let (locals, _clipped) = tt_decomposition(state_tensor, max_rank, use_svd);
        let mut m = MPS::from_tensors(locals, Some(false));
        m.center = Some(m.length - 1);
        m
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
}
