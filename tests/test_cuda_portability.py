import numpy as np
import pytest
import torch

from tensor_network.algorithms.gmps import prepend_labels, train_gmps
from tensor_network.algorithms.dyn_feature_selection_OEE import OEE_variation_one_qubit_measurement
from tensor_network.mps.modules import MPS, MPSType
from tensor_network.tensor_gates.functional import apply_gate
from tensor_network.utils.data import split_classification_dataset
from tensor_network.utils.devices import (
    as_numpy,
    get_torch_device,
    linalg_work_device,
    torch_generator_for,
)
from tensor_network.utils.tensors import zeros_state


def test_get_torch_device_explicit_cpu():
    assert get_torch_device("cpu") == torch.device("cpu")


def test_get_torch_device_explicit_cuda_if_available():
    if torch.cuda.is_available():
        assert get_torch_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA was requested"):
            get_torch_device("cuda")


def test_linalg_work_device_rules():
    assert linalg_work_device(torch.device("cpu")) == torch.device("cpu")
    assert linalg_work_device(torch.device("cuda")) == torch.device("cuda")
    assert linalg_work_device(torch.device("mps")) == torch.device("cpu")


def test_as_numpy_cpu_tensor_and_numpy_array():
    tensor = torch.tensor([1.0, 2.0])
    array = np.array([3.0, 4.0])
    assert np.allclose(as_numpy(tensor), np.array([1.0, 2.0]))
    assert as_numpy(array) is array


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_as_numpy_cuda_tensor():
    tensor = torch.tensor([1.0, 2.0], device="cuda")
    assert np.allclose(as_numpy(tensor), np.array([1.0, 2.0]))


def test_apply_gate_keeps_state_device_with_cpu_gate():
    device = get_torch_device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    state = zeros_state(num_qubits=1, dtype=torch.complex64, device=device)
    gate = torch.eye(2, dtype=torch.complex64)

    out = apply_gate(quantum_state=state, gate=gate, target_qubit=0)

    assert out.device == state.device
    assert torch.allclose(out, state)


def test_split_classification_dataset_keeps_device():
    device = get_torch_device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = torch.arange(24, device=device).reshape(12, 2)
    targets = torch.tensor([0, 1] * 6, dtype=torch.long, device=device)

    train_x, train_y, test_x, test_y = split_classification_dataset(
        data, targets, ratio=0.5, shuffle=True
    )

    assert train_x.device == data.device
    assert train_y.device == targets.device
    assert test_x.device == data.device
    assert test_y.device == targets.device


def test_prepend_labels_moves_labels_to_image_device():
    device = get_torch_device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    raw_images = torch.zeros(2, 28 * 28, device=device)
    labels = torch.tensor([1, 2], dtype=torch.long)

    out = prepend_labels(raw_images, labels)

    assert out.device == raw_images.device
    assert out.dtype == raw_images.dtype
    assert out.shape == (2, 4 + 28 * 28)


def test_mps_projection_moves_cpu_projection_tensor_to_mps_device_and_dtype():
    device = get_torch_device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mps = MPS(
        length=2,
        physical_dim=2,
        virtual_dim=2,
        mps_type=MPSType.Open,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    projection_state = torch.tensor([1.0, 0.0], dtype=torch.float64)

    projected = mps.project_one_qubit(0, projection_state)

    assert all(tensor.device.type == mps.device.type for tensor in projected.local_tensors)
    assert all(tensor.dtype == mps.dtype for tensor in projected.local_tensors)


def test_oee_measurement_moves_cpu_features_to_mps_device_and_dtype():
    device = get_torch_device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    mps = MPS(
        length=3,
        physical_dim=2,
        virtual_dim=2,
        mps_type=MPSType.Open,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )
    feature = torch.tensor(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        dtype=torch.float64,
    )

    oee_changes = OEE_variation_one_qubit_measurement(
        mps, feature, progress_bar_kwargs={"disable": True}
    )

    assert oee_changes.device.type == mps.device.type
    assert oee_changes.dtype == mps.dtype
    assert oee_changes.shape == (3,)


def test_torch_generator_for_rules():
    assert torch_generator_for(torch.device("mps")) is None
    assert torch_generator_for(torch.device("cpu"), seed=123).device.type == "cpu"
    if torch.cuda.is_available():
        assert torch_generator_for(torch.device("cuda"), seed=123).device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tiny_gmps_training_runs_on_cuda():
    device = torch.device("cuda")
    samples = torch.rand(4, 3, 2, device=device)
    samples = samples / samples.norm(dim=2, keepdim=True)
    mps = MPS(
        length=3,
        physical_dim=2,
        virtual_dim=2,
        mps_type=MPSType.Open,
        dtype=torch.float32,
        device=device,
        requires_grad=False,
    )

    losses, trained = train_gmps(
        samples=samples,
        batch_size=2,
        mps=mps,
        sweep_times=1,
        lr=1e-3,
        device=device,
        enable_tsgo=False,
        progress_bar_kwargs={"disable": True},
    )

    assert losses.device.type == "cuda"
    assert trained.device.type == "cuda"
