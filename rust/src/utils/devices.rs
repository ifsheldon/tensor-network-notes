//! Device helpers mirroring the notebook-backed Python utility module.

use tch::{Cuda, Device, Tensor};

use crate::types::DevicePreference;

/// Resolve the Torch device used by examples and tests.
pub fn get_torch_device(prefer: DevicePreference) -> Device {
    match prefer {
        DevicePreference::Auto => {
            if Cuda::is_available() {
                Device::Cuda(0)
            } else {
                Device::Cpu
            }
        }
        DevicePreference::Cuda => {
            assert!(
                Cuda::is_available(),
                "CUDA was requested, but tch::Cuda::is_available() is false"
            );
            Device::Cuda(0)
        }
        DevicePreference::Mps => Device::Mps,
        DevicePreference::Cpu => Device::Cpu,
    }
}

/// Return the device to use for linalg calls with known MPS backend gaps.
pub fn linalg_work_device(device: Device) -> Device {
    match device {
        Device::Mps => Device::Cpu,
        other => other,
    }
}

/// Convert a tensor to a detached CPU tensor suitable for NumPy-style boundaries.
pub fn as_cpu_tensor(x: &Tensor) -> Tensor {
    x.detach().to_device(Device::Cpu)
}
