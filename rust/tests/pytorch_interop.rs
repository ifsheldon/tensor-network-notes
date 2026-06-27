#![cfg(feature = "python-interop")]

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use tch::Tensor;
use tensor_network_code::algorithms::gmps::eval_nll;
use tensor_network_code::feature_mapping::cossin_feature_map;
use tensor_network_code::mps::MPS;
use tensor_network_code::tensor_gates::functional::apply_gate;
use tensor_network_code::utils::data::{
    ImageLoadOptions, ImagePreprocess, ImageSubset, load_fashion_mnist_images_from_cache,
    load_mnist_images_from_cache, split_classification_dataset,
};

const MNIST_ROWS: usize = 28;
const MNIST_COLS: usize = 28;
const MNIST_IMAGE_PIXELS: usize = MNIST_ROWS * MNIST_COLS;

fn rust_impl(x: &Tensor) -> Tensor {
    x.sin() + 2.0
}

fn python_test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn with_python_test(f: impl for<'py> FnOnce(Python<'py>) -> PyResult<()>) -> PyResult<()> {
    let _guard = python_test_lock()
        .lock()
        .expect("Python interop test lock was poisoned");
    Python::with_gil(f)
}

fn run_fixture<'py>(py: Python<'py>, code: &str) -> PyResult<Bound<'py, PyDict>> {
    let locals = PyDict::new(py);
    let source =
        CString::new(code).expect("Python fixture source must not contain interior NUL bytes");
    py.run(source.as_c_str(), None, Some(&locals))?;
    Ok(locals)
}

fn extract_torch_tensor(ob: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
    // SAFETY: `Bound<PyAny>` holds a live PyObject while the GIL is held, and `pyobject_unpack`
    // checks that the object is a wrapped torch tensor before returning `Some`.
    let tensor = unsafe { Tensor::pyobject_unpack(ptr) }
        .map_err(|err| PyErr::new::<PyValueError, _>(format!("{err:?}")))?;
    tensor.ok_or_else(|| {
        let type_ = ob.get_type();
        PyErr::new::<PyTypeError, _>(format!("expected a torch.Tensor, got {type_}"))
    })
}

fn assert_tensors_allclose(name: &str, actual: &Tensor, expected: &Tensor, rtol: f64, atol: f64) {
    assert_eq!(actual.size(), expected.size(), "{name} shape mismatch");
    assert_eq!(actual.kind(), expected.kind(), "{name} kind mismatch");
    let max_abs_diff = (actual - expected).abs().max().double_value(&[]);
    assert!(
        actual.allclose(expected, rtol, atol, false),
        "{name} mismatch; max_abs_diff={max_abs_diff}\nactual={actual:?}\nexpected={expected:?}",
    );
}

fn tiny_train_images() -> Vec<u8> {
    tiny_images(6, 11)
}

fn tiny_test_images() -> Vec<u8> {
    tiny_images(4, 97)
}

fn tiny_images(samples: usize, offset: u8) -> Vec<u8> {
    (0..samples * MNIST_IMAGE_PIXELS)
        .map(|idx| offset.wrapping_add((idx % 251) as u8))
        .collect()
}

fn write_u32_be(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_be_bytes());
}

fn write_idx_images(path: &Path, samples: usize, data: &[u8]) -> std::io::Result<()> {
    assert_eq!(data.len(), samples * MNIST_IMAGE_PIXELS);
    let mut bytes = Vec::with_capacity(16 + data.len());
    write_u32_be(&mut bytes, 2051);
    write_u32_be(&mut bytes, samples as u32);
    write_u32_be(&mut bytes, MNIST_ROWS as u32);
    write_u32_be(&mut bytes, MNIST_COLS as u32);
    bytes.extend_from_slice(data);
    fs::write(path, bytes)
}

fn write_idx_labels(path: &Path, labels: &[u8]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(8 + labels.len());
    write_u32_be(&mut bytes, 2049);
    write_u32_be(&mut bytes, labels.len() as u32);
    bytes.extend_from_slice(labels);
    fs::write(path, bytes)
}

fn write_mnist_style_files(dir: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    let train_images = tiny_train_images();
    let test_images = tiny_test_images();
    write_idx_images(&dir.join("train-images-idx3-ubyte"), 6, &train_images)?;
    write_idx_labels(&dir.join("train-labels-idx1-ubyte"), &[0, 1, 2, 3, 4, 1])?;
    write_idx_images(&dir.join("t10k-images-idx3-ubyte"), 4, &test_images)?;
    write_idx_labels(&dir.join("t10k-labels-idx1-ubyte"), &[1, 2, 9, 1])?;
    Ok(())
}

fn create_mnist_style_cache() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_mnist_style_files(tmp.path()).expect("write direct tch cache");
    write_mnist_style_files(&tmp.path().join("MNIST").join("raw"))
        .expect("write torchvision MNIST cache");
    write_mnist_style_files(&tmp.path().join("FashionMNIST").join("raw"))
        .expect("write torchvision FashionMNIST cache");
    tmp
}

#[test]
fn pytorch_tensor_interop_matches_python_reference() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import torch

torch.manual_seed(0)

with torch.inference_mode():
    x = torch.randn(2, 3, dtype=torch.float32).detach().cpu().contiguous()
    expected = (torch.sin(x) + 2.0).detach().cpu().contiguous()
"#,
        )?;

        let x = extract_torch_tensor(&locals.get_item("x")?.expect("Python fixture missing x"))?;
        let expected = extract_torch_tensor(
            &locals
                .get_item("expected")?
                .expect("Python fixture missing expected"),
        )?;

        let actual = rust_impl(&x);

        assert_eq!(actual.size(), expected.size());
        assert_eq!(actual.kind(), expected.kind());

        let rtol = 1e-5;
        let atol = 1e-8;
        let max_abs_diff = (&actual - &expected).abs().max().double_value(&[]);

        assert!(
            actual.allclose(&expected, rtol, atol, false),
            "tensor mismatch; max_abs_diff={max_abs_diff}\nactual={actual:?}\nexpected={expected:?}",
        );

        let roundtrip = locals
            .get_item("expected")?
            .expect("Python fixture missing expected");
        let roundtrip = extract_torch_tensor(&roundtrip)?;
        assert!(
            roundtrip.allclose(&expected, rtol, atol, false),
            "roundtrip tensor changed after extraction"
        );

        Ok(())
    })
}

#[test]
fn load_mnist_images_from_cache_matches_python_unit_range_oracle() -> PyResult<()> {
    let tmp = create_mnist_style_cache();
    with_python_test(|py| {
        let cache_path = tmp.path().to_string_lossy();
        let locals = run_fixture(
            py,
            &format!(
                r#"
import sys
sys.path.insert(0, "..")
from tensor_network.utils.data import load_mnist_images

cache_path = {cache_path:?}
expected_images, expected_labels = load_mnist_images(
    cache_path=cache_path,
    num=3,
    from_subset="all",
    shuffle=False,
    normalization=False,
    classes=[1, 2],
    return_labels=True,
)
expected_images = expected_images.detach().cpu().contiguous()
expected_labels = expected_labels.detach().cpu().contiguous()
"#,
            ),
        )?;
        let expected_images =
            extract_torch_tensor(&locals.get_item("expected_images")?.expect("missing images"))?;
        let expected_labels =
            extract_torch_tensor(&locals.get_item("expected_labels")?.expect("missing labels"))?;

        let classes = [1, 2];
        let actual = load_mnist_images_from_cache(
            tmp.path(),
            ImageLoadOptions {
                subset: ImageSubset::All,
                num: Some(3),
                classes: Some(&classes),
                ..Default::default()
            },
        )
        .expect("load Rust MNIST fixture");

        assert_tensors_allclose("MNIST images", &actual.images, &expected_images, 1e-6, 1e-8);
        assert_tensors_allclose("MNIST labels", &actual.labels, &expected_labels, 0.0, 0.0);
        Ok(())
    })
}

#[test]
fn load_mnist_images_from_cache_matches_python_standardized_oracle() -> PyResult<()> {
    let tmp = create_mnist_style_cache();
    with_python_test(|py| {
        let cache_path = tmp.path().to_string_lossy();
        let locals = run_fixture(
            py,
            &format!(
                r#"
import sys
sys.path.insert(0, "..")
from tensor_network.utils.data import load_mnist_images

cache_path = {cache_path:?}
expected_images, expected_labels = load_mnist_images(
    cache_path=cache_path,
    num=4,
    from_subset="train",
    shuffle=False,
    normalization=True,
    classes=None,
    return_labels=True,
)
expected_images = expected_images.detach().cpu().contiguous()
expected_labels = expected_labels.detach().cpu().contiguous()
"#,
            ),
        )?;
        let expected_images =
            extract_torch_tensor(&locals.get_item("expected_images")?.expect("missing images"))?;
        let expected_labels =
            extract_torch_tensor(&locals.get_item("expected_labels")?.expect("missing labels"))?;

        let actual = load_mnist_images_from_cache(
            tmp.path(),
            ImageLoadOptions {
                subset: ImageSubset::Train,
                num: Some(4),
                preprocess: ImagePreprocess::mnist_standardized(),
                ..Default::default()
            },
        )
        .expect("load Rust MNIST fixture");

        assert_tensors_allclose(
            "standardized MNIST images",
            &actual.images,
            &expected_images,
            1e-5,
            1e-6,
        );
        assert_tensors_allclose(
            "standardized MNIST labels",
            &actual.labels,
            &expected_labels,
            0.0,
            0.0,
        );
        Ok(())
    })
}

#[test]
fn load_fashion_mnist_images_from_cache_matches_python_unit_range_oracle() -> PyResult<()> {
    let tmp = create_mnist_style_cache();
    with_python_test(|py| {
        let cache_path = tmp.path().to_string_lossy();
        let locals = run_fixture(
            py,
            &format!(
                r#"
import sys
sys.path.insert(0, "..")
from torch.utils import data
from tensor_network.utils.data import get_fashion_mnist_datasets

cache_path = {cache_path:?}
_, test_set = get_fashion_mnist_datasets(cache_path)
expected_images, expected_labels = next(iter(data.DataLoader(test_set, batch_size=2, shuffle=False)))
expected_images = expected_images.detach().cpu().contiguous()
expected_labels = expected_labels.detach().cpu().contiguous()
"#,
            ),
        )?;
        let expected_images =
            extract_torch_tensor(&locals.get_item("expected_images")?.expect("missing images"))?;
        let expected_labels =
            extract_torch_tensor(&locals.get_item("expected_labels")?.expect("missing labels"))?;

        let actual = load_fashion_mnist_images_from_cache(
            tmp.path(),
            ImageLoadOptions {
                subset: ImageSubset::Test,
                num: Some(2),
                ..Default::default()
            },
        )
        .expect("load Rust Fashion-MNIST fixture");

        assert_tensors_allclose(
            "Fashion-MNIST images",
            &actual.images,
            &expected_images,
            1e-6,
            1e-8,
        );
        assert_tensors_allclose(
            "Fashion-MNIST labels",
            &actual.labels,
            &expected_labels,
            0.0,
            0.0,
        );
        Ok(())
    })
}

#[test]
fn cossin_feature_map_matches_python_export() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import sys
sys.path.insert(0, "..")
import torch
from tensor_network.feature_mapping import cossin_feature_map

samples = torch.tensor([[0.0, 0.25, 1.0], [0.5, 0.75, 0.1]], dtype=torch.float32)
expected = cossin_feature_map(samples, theta=0.7).detach().cpu().contiguous()
"#,
        )?;

        let samples = extract_torch_tensor(&locals.get_item("samples")?.expect("missing samples"))?;
        let expected =
            extract_torch_tensor(&locals.get_item("expected")?.expect("missing expected"))?;
        let actual = cossin_feature_map(&samples, 0.7, true);

        assert_eq!(actual.size(), expected.size());
        assert!(
            actual.allclose(&expected, 1e-5, 1e-8, false),
            "feature map mismatch\nactual={actual:?}\nexpected={expected:?}"
        );
        Ok(())
    })
}

#[test]
fn apply_gate_matches_python_export() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import sys
sys.path.insert(0, "..")
import torch
from tensor_network.utils.tensors import zeros_state
from tensor_network.tensor_gates.functional import apply_gate

state = zeros_state(num_qubits=3, dtype=torch.complex64)
gate = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
expected = apply_gate(quantum_state=state, gate=gate, target_qubit=1).detach().cpu().contiguous()
"#,
        )?;

        let state = extract_torch_tensor(&locals.get_item("state")?.expect("missing state"))?;
        let gate = extract_torch_tensor(&locals.get_item("gate")?.expect("missing gate"))?;
        let expected =
            extract_torch_tensor(&locals.get_item("expected")?.expect("missing expected"))?;
        let actual = apply_gate(&state, &gate, &[1], &[]);

        assert_eq!(actual.size(), expected.size());
        assert!(
            actual.allclose(&expected, 1e-5, 1e-8, false),
            "apply_gate mismatch\nactual={actual:?}\nexpected={expected:?}"
        );
        Ok(())
    })
}

#[test]
fn mps_global_tensor_matches_python_export() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import sys
sys.path.insert(0, "..")
import torch
from tensor_network.mps.functional import calc_global_tensor_by_tensordot

torch.manual_seed(3)
t0 = torch.randn(1, 2, 3, dtype=torch.float32)
t1 = torch.randn(3, 2, 4, dtype=torch.float32)
t2 = torch.randn(4, 2, 1, dtype=torch.float32)
expected = calc_global_tensor_by_tensordot([t0, t1, t2]).detach().cpu().contiguous()
"#,
        )?;

        let t0 = extract_torch_tensor(&locals.get_item("t0")?.expect("missing t0"))?;
        let t1 = extract_torch_tensor(&locals.get_item("t1")?.expect("missing t1"))?;
        let t2 = extract_torch_tensor(&locals.get_item("t2")?.expect("missing t2"))?;
        let expected =
            extract_torch_tensor(&locals.get_item("expected")?.expect("missing expected"))?;
        let actual = MPS::from_tensors(vec![t0, t1, t2]).global_tensor();

        assert_eq!(actual.size(), expected.size());
        assert!(
            actual.allclose(&expected, 1e-5, 1e-8, false),
            "MPS global tensor mismatch\nactual={actual:?}\nexpected={expected:?}"
        );
        Ok(())
    })
}

#[test]
fn gmps_eval_nll_matches_python_export() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import sys
sys.path.insert(0, "..")
import torch
from tensor_network.mps.modules import MPS
from tensor_network.algorithms.gmps import eval_nll

torch.manual_seed(7)
t0 = torch.randn(1, 2, 3, dtype=torch.float32)
t1 = torch.randn(3, 2, 2, dtype=torch.float32)
t2 = torch.randn(2, 2, 1, dtype=torch.float32)
samples = torch.rand(5, 3, 2, dtype=torch.float32)
mps = MPS(mps_tensors=[t0, t1, t2])
mps._center = 1
expected = eval_nll(samples=samples, mps=mps, device=torch.device("cpu"), return_avg=False).detach().cpu().contiguous()
"#,
        )?;

        let t0 = extract_torch_tensor(&locals.get_item("t0")?.expect("missing t0"))?;
        let t1 = extract_torch_tensor(&locals.get_item("t1")?.expect("missing t1"))?;
        let t2 = extract_torch_tensor(&locals.get_item("t2")?.expect("missing t2"))?;
        let samples = extract_torch_tensor(&locals.get_item("samples")?.expect("missing samples"))?;
        let expected =
            extract_torch_tensor(&locals.get_item("expected")?.expect("missing expected"))?;
        let mut mps = MPS::from_tensors(vec![t0, t1, t2]);
        mps.set_center(Some(1));
        let actual = eval_nll(&samples, &mps, tch::Device::Cpu, false);

        assert_eq!(actual.size(), expected.size());
        assert!(
            actual.allclose(&expected, 1e-5, 1e-8, false),
            "GMPS eval_nll mismatch\nactual={actual:?}\nexpected={expected:?}"
        );
        Ok(())
    })
}

#[test]
fn split_classification_dataset_matches_python_export() -> PyResult<()> {
    with_python_test(|py| {
        let locals = run_fixture(
            py,
            r#"
import sys
sys.path.insert(0, "..")
import torch
from tensor_network.utils.data import split_classification_dataset

data = torch.arange(24, dtype=torch.float32).reshape(12, 2)
targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long)
expected = split_classification_dataset(data, targets, ratio=0.25, shuffle=False)
"#,
        )?;

        let data = extract_torch_tensor(&locals.get_item("data")?.expect("missing data"))?;
        let targets = extract_torch_tensor(&locals.get_item("targets")?.expect("missing targets"))?;
        let expected = locals.get_item("expected")?.expect("missing expected");
        let actual = split_classification_dataset(&data, &targets, 0.25, false);
        let actual = [actual.0, actual.1, actual.2, actual.3];

        for (idx, actual) in actual.iter().enumerate() {
            let expected_i = expected.get_item(idx)?;
            let expected_i = extract_torch_tensor(&expected_i)?;
            assert_eq!(actual.size(), expected_i.size(), "shape mismatch at {idx}");
            assert!(
                actual.allclose(&expected_i, 1e-5, 1e-8, false),
                "split output {idx} mismatch\nactual={actual:?}\nexpected={expected_i:?}"
            );
        }
        Ok(())
    })
}
