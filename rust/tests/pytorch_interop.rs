#![cfg(feature = "python-interop")]

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;
use tch::Tensor;
use tensor_network_code::feature_mapping::cossin_feature_map;
use tensor_network_code::mps::MPS;
use tensor_network_code::tensor_gates::functional::apply_gate;

fn rust_impl(x: &Tensor) -> Tensor {
    x.sin() + 2.0
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

#[test]
fn pytorch_tensor_interop_matches_python_reference() -> PyResult<()> {
    Python::with_gil(|py| {
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
fn cossin_feature_map_matches_python_export() -> PyResult<()> {
    Python::with_gil(|py| {
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
    Python::with_gil(|py| {
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
    Python::with_gil(|py| {
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
