**Tutorial: Comparing Python PyTorch Tensors With `tch-rs`**

Goal: call Python PyTorch code from a Rust test, convert the resulting `torch.Tensor` into a `tch::Tensor`, run the Rust implementation, and compare the tensors in Rust.

**Mental Model**

`pyo3-tch` provides `PyTensor`, a small wrapper around `tch::Tensor`. It implements PyO3 conversions so a Python `torch.Tensor` can be extracted into Rust as a `tch::Tensor`, and its implementation shows the lower-level bridge used by this repo.

Do not use `tensor.data_ptr()` for this. A PyTorch tensor is not just memory. It includes dtype, shape, strides, device, storage ownership, and autograd metadata. `pyo3-tch` uses PyTorch’s own Python/C++ tensor bridge instead.

As of the current crate docs, `pyo3-tch 0.24.0` depends on `pyo3 0.24`, `tch 0.24.0`, and `torch-sys 0.24.0`.
This repository uses the patched `tch 0.25.1` fork pinned at `rust/tch-rs`, because the crates.io release was not available and the fork has the PyTorch 2.12.1 update.

**Cargo Setup**

The direct embedding path used by this repository pins `pyo3` and `tch` directly.

```toml
[features]
python-interop = ["dep:pyo3", "dep:tch"]

[dependencies]
pyo3 = { version = "0.24", features = ["auto-initialize"], optional = true }
tch = { path = "tch-rs", features = ["python-extension"], optional = true }
```

In this repository the dependencies are optional normal dependencies behind Cargo feature `python-interop`, so regular Rust tests do not need to build or link libtorch.
The actual experiment uses `tch::Tensor::pyobject_unpack` directly instead of `pyo3-tch::PyTensor`, because `pyo3-tch 0.24.0` enables PyO3's `extension-module` feature and that fails when this crate embeds Python from a Rust test.

Run tests against the same Python environment that has `torch` installed:

```bash
LIBTORCH_USE_PYTORCH=1 PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo test
```

The repository wrapper for the interop experiment is:

```bash
poe rusttest_interop
```

`LIBTORCH_USE_PYTORCH=1` is important: it tells `tch` to link against the active Python PyTorch installation, reducing ABI/version mismatch risk.
On this machine the wrapper embeds `/usr/bin/python3.12`, then prepends the uv environment's `site-packages` to `PYTHONPATH` so `torch` still comes from the pinned project environment.
This avoids a uv standalone Python embedding issue where `_ctypes` is available to the `uv run python` executable but not to the embedded libpython runtime.

**Rust Test Pattern**

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::CString;
use tch::Tensor;

fn rust_impl(x: &Tensor) -> Tensor {
    x.sin() + 2.0
}

fn extract_torch_tensor(ob: &Bound<'_, PyAny>) -> PyResult<Tensor> {
    let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
    let tensor = unsafe { Tensor::pyobject_unpack(ptr) }
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(format!("{err:?}")))?;
    tensor.ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("expected a torch.Tensor"))
}

#[test]
fn rust_matches_python_reference() -> PyResult<()> {
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
        .unwrap();

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

        Ok(())
    })
}
```

**Agent Checklist**

1. Pin `pyo3` and `tch` to compatible versions.
2. If using `pyo3-tch`, use its `tch` re-export; otherwise make the direct `tch` dependency exact.
3. Always acquire the Python GIL before touching Python objects.
4. Extract Python tensors with `.extract::<PyTensor>()` or `Tensor::pyobject_unpack`.
5. Use `.detach().cpu().contiguous()` in Python test code unless CUDA, gradients, or strides are part of the behavior under test.
6. Compare `size()`, `kind()`, then `allclose()`.
7. Print `max_abs_diff` on failure.
8. Treat mutable shared tensors carefully: this conversion can share underlying PyTorch storage.

**Important Build Trap**

`pyo3-tch` is primarily designed for Python extension modules. Its crate metadata enables PyO3’s `extension-module` feature, and PyO3 documents that this can make Rust binaries/tests which embed Python fail to link on some platforms.

If that happens, it is probably a build-mode issue, not a tensor-conversion issue. The fallback is to use the same underlying `tch::Tensor::pyobject_unpack` bridge directly in a local test helper, without depending on `pyo3-tch`’s extension-module packaging.

Sources: [`pyo3-tch` crate](https://docs.rs/crate/pyo3-tch/latest), [`pyo3-tch` source](https://docs.rs/crate/pyo3-tch/latest/source/src/lib.rs), [PyO3 embedding/build notes](https://pyo3.rs/main/building-and-distribution), [`tch-rs`](https://github.com/LaurentMazare/tch-rs).
