use crate::utils::einsum::named_einsum;
use tch::{Device, IndexOp, Kind, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MPSType {
    Open,
    Periodic,
}

pub fn get_mps_type(mps: &[Tensor]) -> MPSType {
    if mps.first().unwrap().size()[0] == 1 && mps.last().unwrap().size()[2] == 1 {
        MPSType::Open
    } else {
        MPSType::Periodic
    }
}

pub fn gen_random_mps_tensors(
    length: i64,
    physical_dim: i64,
    virtual_dim: i64,
    mps_type: MPSType,
    kind: Kind,
    device: Device,
) -> Vec<Tensor> {
    match mps_type {
        MPSType::Open => {
            let mut out = Vec::with_capacity(length as usize);
            out.push(Tensor::randn(
                [1, physical_dim, virtual_dim],
                (kind, device),
            ));
            for _ in 0..(length - 2) {
                out.push(Tensor::randn(
                    [virtual_dim, physical_dim, virtual_dim],
                    (kind, device),
                ));
            }
            out.push(Tensor::randn(
                [virtual_dim, physical_dim, 1],
                (kind, device),
            ));
            out
        }
        MPSType::Periodic => {
            let mut out = Vec::with_capacity(length as usize);
            for _ in 0..length {
                out.push(Tensor::randn(
                    [virtual_dim, physical_dim, virtual_dim],
                    (kind, device),
                ));
            }
            out
        }
    }
}

pub fn calc_global_tensor_by_contract(mps: &[Tensor]) -> Tensor {
    // Build a single named-einsum that joins all 3-d tensors along virtual bonds.
    let n = mps.len();
    let mut inputs: Vec<String> = Vec::with_capacity(n);
    let mut outputs: Vec<String> = Vec::new();
    let mut specs: Vec<Vec<String>> = Vec::new();
    for i in 0..n {
        let labels = vec![format!("t{}0", i), format!("t{}1", i), format!("t{}2", i)];
        inputs.push(labels.join(" "));
        specs.push(labels);
    }
    let mps_type = get_mps_type(mps);
    if mps_type == MPSType::Periodic {
        // contract right of i with left of i+1 and last right with first left
        // Output keeps only all physical dims t{i}1
        outputs = (0..n).map(|i| format!("t{}1", i)).collect();
    } else {
        // Open: keep first left and last right too
        outputs.push("t00".to_string());
        outputs.extend((0..n).map(|i| format!("t{}1", i)));
        outputs.push(format!("t{}2", n - 1));
    }
    let spec = format!("{} -> {}", inputs.join(", "), outputs.join(" "));
    let owned: Vec<Tensor> = mps.iter().map(|t| t.shallow_clone()).collect();
    named_einsum(&spec, &owned).squeeze()
}

pub fn calculate_mps_norm_factors(mps: &[Tensor], efficient_open: bool) -> Tensor {
    assert!(!mps.is_empty());
    let conj: Vec<Tensor> = mps.iter().map(|t| t.conj()).collect();
    let n = mps.len();
    let device = conj[0].device();
    let kind = conj[0].kind();
    match get_mps_type(mps) {
        MPSType::Open => {
            let mut v = Tensor::ones([1, 1], (kind, device)); // a b
            let norms = Tensor::empty([n as i64], (kind, device));
            if efficient_open {
                for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                    v = Tensor::einsum("ab,aix->bix", &[v, _c.shallow_clone()], None::<Vec<i64>>);
                    v = Tensor::einsum("bix,biy->xy", &[v, _m.shallow_clone()], None::<Vec<i64>>);
                    let nf = v.norm();
                    v = &v / &nf;
                    norms.i((i as i64,)).copy_(&nf);
                }
            } else {
                for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                    v = Tensor::einsum(
                        "ab,aix,biy->xy",
                        &[v, _c.shallow_clone(), _m.shallow_clone()],
                        None::<Vec<i64>>,
                    );
                    let nf = v.norm();
                    v = &v / &nf;
                    norms.i((i as i64,)).copy_(&nf);
                }
            }
            norms
        }
        MPSType::Periodic => {
            let vdim = mps[0].size()[0];
            let norms = Tensor::empty([n as i64], (kind, device));
            let mut v = Tensor::eye(vdim * vdim, (kind, device)).view([vdim, vdim, vdim, vdim]);
            for (i, (_c, _m)) in conj.iter().zip(mps.iter()).enumerate() {
                v = Tensor::einsum(
                    "uvap,adb,pdq->uvbq",
                    &[v, _c.shallow_clone(), _m.shallow_clone()],
                    None::<Vec<i64>>,
                );
                let nf = v.norm();
                v = &v / &nf;
                norms.i((i as i64,)).copy_(&nf);
            }
            let final_nf = Tensor::einsum("acac->", &[v], None::<Vec<i64>>);
            let last = norms.i((n as i64 - 1,)) * final_nf;
            norms.i((n as i64 - 1,)).copy_(&last);
            norms
        }
    }
}

pub fn normalize_mps(mps: &[Tensor]) -> Vec<Tensor> {
    let norms = calculate_mps_norm_factors(mps, true);
    let factors = 1.0f64 / norms.sqrt();
    let n = mps.len();
    let mut out: Vec<Tensor> = Vec::with_capacity(n);
    for (i, t) in mps.iter().enumerate() {
        let f = factors.double_value(&[i as i64]);
        out.push(t * f);
    }
    out
}

pub fn calc_inner_product(mps0: &[Tensor], mps1: &[Tensor]) -> Tensor {
    assert_eq!(mps0.len(), mps1.len());
    let n = mps0.len();
    let kind = mps0[0].kind();
    let device = mps0[0].device();
    let v0 = Tensor::eye(mps0[0].size()[0], (kind, device));
    let v1 = Tensor::eye(mps1[0].size()[0], (kind, device));
    let mut v = Tensor::einsum("ab,xy->axby", &[v0, v1], None::<Vec<i64>>);
    let factors = Tensor::empty([(n as i64) + 1], (kind, device));
    for (i, (m0, m1)) in mps0.iter().zip(mps1.iter()).enumerate() {
        v = Tensor::einsum(
            "uvap,adb,pdq->uvbq",
            &[v, m0.conj(), m1.shallow_clone()],
            None::<Vec<i64>>,
        );
        let nf = v.norm();
        v = &v / &nf;
        factors.i((i as i64,)).copy_(&nf);
    }
    let last = if v.numel() > 1 {
        Tensor::einsum("acac->", &[v], None::<Vec<i64>>)
    } else {
        v.reshape([1]).i(0)
    };
    factors.i((n as i64,)).copy_(&last);
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_tensor_shape_open() {
        let mps = gen_random_mps_tensors(3, 2, 3, MPSType::Open, Kind::Float, Device::Cpu);
        let g = calc_global_tensor_by_contract(&mps);
        assert_eq!(g.size(), vec![2, 2, 2]);
    }
}
