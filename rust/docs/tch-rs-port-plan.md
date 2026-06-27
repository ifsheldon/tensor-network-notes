# Rust `tch-rs` Port Plan

## Summary

Port the non-MLX `tensor_network` library into the Rust crate under `rust/`, using `tch-rs` as the tensor backend and Python/PyTorch only as a test oracle.

The first implementation milestone is `MPS`, not `MPSParameter`.
`MPS` is tensor-like, not parameter-like: it owns or carries `Vec<tch::Tensor>`, participates in autograd when its inner tensors do, and knows nothing about `VarStore` or `nn::Path`.

Excluded from the port: `tensor_network/mlx/**`, nbdev generated metadata such as `_modidx.py`, and Python-only reference import helpers such as `setup_ref_code_import.py`.

## Key Decisions

- Use the Rust type name `MPS`, not `Mps`, and `MPSType`, not `MpsType`.
- Add a local `#[allow(clippy::upper_case_acronyms)]` only where needed.
- Do not use Python trailing-underscore naming.
- Mutating methods take `&mut self`; when both forms exist, use pairs like `normalize` and `normalized`, `center_orthogonalize` and `center_orthogonalized`, `set_device` and `to_device`, `set_kind` and `to_kind`.
- `MPS` is tensor-like and has no `VarStore`, no `nn::Path`, and no parameter registration constructors.
- Defer `MPSParameter`.
- Reserve the future design that `MPSParameter` is parameter-like, constructed from `&nn::Path`, and exposes `as_mps() -> MPS` by shallow-cloning tensor handles.
- Keep the Rust crate Rust-only.
- Keep PyO3 only behind the `python-interop` test feature for PyTorch tensor extraction and parity tests.
- Bump the local `tch-rs` fork from `safetensors = "0.3.0"` to latest compatible `0.8.x` as an early focused experiment.
- Patch `tch-rs` if its safetensors wrapper API needs adjustment after the bump.
- Preserve Python MPS artifact compatibility exactly: tensor keys `"0"`, `"1"`, ... and scalar key `"center"`, where `-1` means no center.
- Keep Python semantic checks as Rust `assert!` or `assert_eq!`, especially for dimensions, ranges, device compatibility, dtype compatibility, algorithm preconditions, and numerical sanity checks.
- Omit Python type checks when Rust's type system makes the invalid case impossible.
- Public Rust APIs return `Result<T, TensorNetworkError>` for IO and fallible `tch` operations, while semantic precondition failures remain assertions.

## Assertion Porting Taxonomy

- Type checks: Python checks like `isinstance(x, torch.Tensor)`, `isinstance(indices, (List, torch.Tensor))`, and `isinstance(target_qubit, (int, list))` should become Rust signatures, enums, or typed constructors when possible.
- Option exclusivity and required arguments: checks such as "exactly one of `num_qubits` and `init_qubit_state`" or "provide either `params_vec` or scalar rotation parameters" should stay as assertions unless the Rust API can represent them with separate constructors or enum variants.
- Shape and rank checks: assertions on `ndim`, exact tensor shape, square matrices, matching batch sizes, matching feature counts, and gate tensor rank must stay as `assert!` or `assert_eq!`.
- Quantum structure checks: state tensors with all dimensions equal to 2, gate tensors with `2 * num_qubits` dimensions, matrix gates with shape `(2^q, 2^q)`, non-overlapping control and target qubits, and valid qubit index ranges must stay as assertions.
- Dtype and device compatibility checks: assertions that tensors share dtype/device, that dtypes are float or complex, or that labels are integer tensors must stay unless encoded in a wrapper type later.
- Value-domain checks: positive dimensions, valid ratios, thresholds, time steps, tau ranges, sample values in `[0, 1]`, probabilities in `[0, 1]`, and class labels in range must stay as assertions.
- Enum-like option checks: Python string options such as gate pattern, kernel name, Pauli direction, orthogonalization mode, generation order, and device preference should become Rust enums; once typed, the old membership assertions disappear.
- Collection consistency checks: non-empty lists, equal list lengths, unique indices, selected feature counts, dataset divisibility by batch size, and matching numbers of Hamiltonians and positions must stay as assertions.
- Algorithm state checks: MPS center must exist before center-dependent algorithms, MPS must be open for GMPS training, center movement invariants, orthogonality checks, and reference-comparison asserts must stay.
- Numerical sanity checks: NaN checks after decompositions, allclose checks in reference/debug paths, Hermitian/symmetric checks, probability normalization assumptions, and rank/truncation checks must stay.
- Unsupported and unreachable branches: Python `NotImplementedError`, `ValueError`, and `Exception("Unreachable")` should become Rust `panic!` or `unreachable!` when the branch is impossible by construction, and `Result` errors only for external IO or truly recoverable runtime failures.

## Implementation Plan

### 1. Rust Crate Foundation

- Make `tch` a normal Rust dependency in `rust/Cargo.toml`.
- Keep `pyo3` optional and use `python-interop = ["dep:pyo3", "tch/python-extension"]`.
- Add `TensorNetworkError`, `Result<T>`, and small typed enums: `DevicePreference`, `OrthogonalizationMode`, `GatePattern`, `Kernel`, and `MPSType`.
- Add a module tree mirroring the Python package: `utils`, `feature_mapping`, `tensor_gates`, `quantum_state`, `mps`, `algorithms`, and `networks`.
- Update README Rust notes with the new split: Rust crate is implementation code, Python interop is test-only, and `MPSParameter` is deferred.

### 2. Safetensors Bump in `tch-rs`

- Update the local `rust/tch-rs` fork dependency from `safetensors = "0.3.0"` to the latest compatible `0.8.x` release.
- Regenerate `rust/Cargo.lock` from the `rust/` crate after the dependency change.
- Compile `tch-rs` through the top-level Rust crate with `uv run poe rusttest` or `cd rust && cargo test`.
- If the `safetensors` API changed, patch `rust/tch-rs/src/tensor/safetensors.rs` while preserving the existing public `tch::Tensor::read_safetensors` and `tch::Tensor::write_safetensors` APIs.
- Add or update `tch-rs` safetensors tests to cover scalar tensors, non-contiguous save errors, ordinary floating tensors, and all dtype mappings supported by both `tch::Kind` and `safetensors::Dtype`.
- Keep complex tensor support as an explicit discovery item during the bump because Python MPS artifacts are expected to be real-valued first, while gate and state tensors may be complex later.
- After the bump, run `uv run poe rusttest_interop` to confirm the Python-linked `tch-rs` build still works with the pinned PyTorch installation.

### 3. First Milestone: `MPS`

- Implement `MPS { tensors: Vec<Tensor>, center: Option<usize> }` with explicit `shallow_clone()` helpers instead of relying on ambiguous Rust `Clone` semantics.
- Port MPS constructors and IO: `MPS::from_tensors`, `MPS::random`, `MPS::from_state_tensor`, `MPS::save_safetensors`, and `MPS::load_safetensors`.
- Port MPS accessors and metadata: `len`, `is_empty`, `physical_dim`, `virtual_dim`, `mps_type`, `center`, `center_tensor`, `device`, `kind`, `local_tensors`, and `local_tensor`.
- Port MPS operations: `global_tensor`, `norm_factors`, `norm`, `normalize`, `normalized`, `inner_product`, `center_orthogonalize`, `center_orthogonalized`, `center_normalize`, `set_local_tensor`, `to_device`, `set_device`, `to_kind`, `set_kind`, `set_requires_grad`, `one_body_reduced_density_matrix`, `project_multi_qubits`, and `project_one_qubit`.
- Convert patched Python methods into normal methods: `onsite_entanglement_entropy` and `two_body_reduced_density_matrix`.
- Preserve differentiability for out-of-place transforms.
- Methods like `normalized()` must build derived tensors without detaching, so gradients can flow back to registered leaf tensors that were used to build the original `MPS`.
- Treat mutating structural methods as algebraic state updates.
- Document that mutating structural methods should not replace registered leaf tensors inside trainable model forward passes until `MPSParameter` exists.

### 4. Core Tensor, Gate, and State APIs

- Port utility modules needed by later code: tensor construction, outer products, normalization, rescaling, device resolution, linalg work-device fallback, dtype promotion, gate matrix/tensor reshaping, and validation.
- Port feature maps: `cossin_feature_map`, `feature_map_to_qubit_state`, and `linear_mapping`.
- Port tensor gates: `apply_gate`, batched and nonbatched paths, Kronecker product, Pauli/spin operators, rotations, control gates, random unitary/gate tensors, and Hamiltonian builders.
- Replace Python inheritance in gate modules with Rust structs and traits.
- Use a `QuantumGate` trait for `forward`.
- Implement `SimpleGate`, `PauliGate`, `ADQCGate`, and `RotateGate`.
- Trainable gate structs may use `&nn::Path`; this does not imply `MPS` uses `Path`.
- Port quantum-state functions: reduced density matrices, observations, onsite and bipartite entropy, projection, and bond energies.
- Preserve CUDA behavior.
- `linalg_work_device` sends only MPS device work to CPU; CUDA and CPU linalg stay on their original device.

### 5. Algorithms and Networks

- Port tensor decomposition: rank-1 decomposition, gradient-based decomposition, matrix unfolding, Tucker decomposition, and reduced matrices.
- Port eigen and ground-state helpers.
- Implement `eigs_power` as a direct tensor algorithm.
- Implement `calc_ground_state` as a dense `tch` implementation for v1 parity on notebook-scale systems; preserve the same API and compare to Python `eigsh` fixtures on small systems.
- Port algorithms depending on MPS: imaginary time evolution, TEBD, OEE feature selection, entanglement-ordered sampling, GMPS training/evaluation/generation/classification.
- Port quantum-kernel and lazy-classifier functions using pure tensor operations; no sklearn dependency is needed for the classifier itself.
- Port trainable networks with standard `tch::nn` patterns: `ADQCNet`, `FCADQCHybridClassifier`, `ADQCRNN`, `ResMPSSimple`, `ADQCTimeEvolution`, and `PolarizationGate`.
- Model constructors receive `&nn::Path`; the top-level caller owns `VarStore`.
- `ResMPSSimple` may directly register its local tensors under its own path, but this is not `MPSParameter`.
- Port data helpers where Rust equivalents are practical.
- Fully port `split_classification_dataset`.
- Load MNIST through `tch::vision::mnist` from an existing cache, returning an error if files are absent.
- Load Iris through `linfa-datasets`.
- Treat Fashion-MNIST as cache-only v1, using the MNIST-format reader if compatible.

## Test Plan

- Add pure Rust unit tests for shape, dtype, device, validation errors, and small deterministic examples.
- Add Python-oracle integration tests behind `python-interop`.
- Python constructs PyTorch tensors and expected outputs.
- Rust extracts PyTorch tensors via `Tensor::pyobject_unpack`.
- Rust implementation output is compared with `allclose`.
- Test `MPS` first: random open/periodic MPS shapes, global tensor equivalence between contract and tensordot paths, norm factors, normalization, inner product, QR/SVD orthogonalization on CPU and CUDA, projection, entropy methods, and gradient flow through `MPS::normalized()`.
- Test safetensors compatibility both ways: Python saves MPS and Rust loads it; Rust saves MPS and Python loads it with `safetensors.torch.load_file`.
- Test each later port with the closest notebook/Python function as oracle, starting with tiny tensors and expanding to CUDA cases where available.
- Required checks before each focused commit: `uv run poe rusttest`, `uv run poe rusttest_interop`, and targeted Python CUDA tests only when the port touches behavior shared with existing notebooks.

## Assumptions and Defaults

- The Rust port lives directly in `rust/src`, not generated from notebooks.
- Numerical parity comes before performance tuning and artifact compatibility beyond MPS safetensors.
- CUDA support is required.
- MPS support is preserved by explicit fallback design but validated by code review unless running on macOS.
- `MPSParameter` is intentionally not implemented in this phase.
- Any trainable model code that needs parameters uses `nn::Path` directly in that model, while `MPS` remains tensor-like.
- The first implementation slice should be crate foundation, safetensors bump, `MPS`, and the minimal utility functions needed to test it.
