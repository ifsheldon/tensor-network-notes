**Tutorial: Comparing Python PyTorch Tensors With `tch-rs` Using `pyo3-tch`**

Goal: call Python PyTorch code from a Rust test, convert the resulting `torch.Tensor` into a `tch::Tensor`, run the Rust implementation, and compare the tensors in Rust.

**Mental Model**

`pyo3-tch` provides `PyTensor`, a small wrapper around `tch::Tensor`. It implements PyO3 conversions so a Python `torch.Tensor` can be extracted into Rust as a `tch::Tensor`.

Do not use `tensor.data_ptr()` for this. A PyTorch tensor is not just memory. It includes dtype, shape, strides, device, storage ownership, and autograd metadata. `pyo3-tch` uses PyTorch’s own Python/C++ tensor bridge instead.

As of the current crate docs, `pyo3-tch 0.24.0` depends on `pyo3 0.24`, `tch 0.24.0`, and `torch-sys 0.24.0`.

**Cargo Setup**

Use the `tch` re-export from `pyo3-tch` to avoid accidentally mixing incompatible `tch` versions.

```toml
[dev-dependencies]
pyo3 = { version = "0.24", features = ["auto-initialize"] }
pyo3-tch = "0.24"
```

Run tests against the same Python environment that has `torch` installed:

```bash
LIBTORCH_USE_PYTORCH=1 PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" cargo test
```

`LIBTORCH_USE_PYTORCH=1` is important: it tells `tch` to link against the active Python PyTorch installation, reducing ABI/version mismatch risk.

**Rust Test Pattern**

```rust
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3_tch::{tch::Tensor, PyTensor};
use std::ffi::CString;

fn rust_impl(x: &Tensor) -> Tensor {
    x.sin() + 2.0
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

        let x: PyTensor = locals
            .get_item("x")?
            .expect("Python fixture missing x")
            .extract()?;

        let expected: PyTensor = locals
            .get_item("expected")?
            .expect("Python fixture missing expected")
            .extract()?;

        let actual = rust_impl(&x.0);

        assert_eq!(actual.size(), expected.0.size());
        assert_eq!(actual.kind(), expected.0.kind());

        let rtol = 1e-5;
        let atol = 1e-8;
        let max_abs_diff = (&actual - &expected.0).abs().max().double_value(&[]);

        assert!(
            actual.allclose(&expected.0, rtol, atol, false),
            "tensor mismatch; max_abs_diff={max_abs_diff}\nactual={actual:?}\nexpected={:?}",
            expected.0
        );

        Ok(())
    })
}
```

**Agent Checklist**

1. Pin `pyo3`, `pyo3-tch`, and `tch` to compatible versions.
2. Use `pyo3_tch::tch::Tensor`, or ensure the project’s direct `tch` dependency exactly matches `pyo3-tch`.
3. Always acquire the Python GIL before touching Python objects.
4. Extract Python tensors with `.extract::<PyTensor>()`.
5. Use `.detach().cpu().contiguous()` in Python test code unless CUDA, gradients, or strides are part of the behavior under test.
6. Compare `size()`, `kind()`, then `allclose()`.
7. Print `max_abs_diff` on failure.
8. Treat mutable shared tensors carefully: this conversion can share underlying PyTorch storage.

**Important Build Trap**

`pyo3-tch` is primarily designed for Python extension modules. Its crate metadata enables PyO3’s `extension-module` feature, and PyO3 documents that this can make Rust binaries/tests which embed Python fail to link on some platforms.

If that happens, it is probably a build-mode issue, not a tensor-conversion issue. The fallback is to use the same underlying `tch::Tensor::pyobject_unpack` bridge directly in a local test helper, without depending on `pyo3-tch`’s extension-module packaging.

Sources: [`pyo3-tch` crate](https://docs.rs/crate/pyo3-tch/latest), [`pyo3-tch` source](https://docs.rs/crate/pyo3-tch/latest/source/src/lib.rs), [PyO3 embedding/build notes](https://pyo3.rs/main/building-and-distribution), [`tch-rs`](https://github.com/LaurentMazare/tch-rs).
