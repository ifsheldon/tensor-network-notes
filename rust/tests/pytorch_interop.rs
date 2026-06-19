#![cfg(feature = "python-interop")]

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;
use tch::Tensor;

fn rust_impl(x: &Tensor) -> Tensor {
    x.sin() + 2.0
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
        let locals = PyDict::new(py);
        let code = CString::new(
            r#"
import torch

torch.manual_seed(0)

with torch.inference_mode():
    x = torch.randn(2, 3, dtype=torch.float32).detach().cpu().contiguous()
    expected = (torch.sin(x) + 2.0).detach().cpu().contiguous()
"#,
        )
        .expect("Python fixture source must not contain interior NUL bytes");

        py.run(code.as_c_str(), None, Some(&locals))?;

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
