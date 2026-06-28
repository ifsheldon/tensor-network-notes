# Rust `tch-rs` Port TODO

## Summary

This document tracks what remains after the initial non-MLX Rust port described in `tch-rs-port-plan.md`.
The Rust crate now has the main module tree, core `MPS`, tensor gates, dense quantum-state helpers, several algorithms, dataset helpers, and the named trainable network structs.
The remaining work is mostly API parity, training/workflow ergonomics, artifact compatibility, and stronger test coverage rather than missing neural-network structs.

## Current Baseline

- Implemented major neural-network structs: `ADQCNet`, `FCADQCHybridClassifier`, `ADQCRNN`, `ResMPSSimple`, `ADQCTimeEvolution`, and `PolarizationGate`.
- Implemented tensor-like `MPS` without `VarStore` or `nn::Path`, as planned.
- Implemented `ADQCGate` and other gate wrappers with `nn::Path` only where the Python code has trainable gate/module parameters.
- Implemented Python/PyTorch interop as a test-only feature through `python-interop`.
- Implemented MPS safetensors read/write with Python-compatible keys, but cross-language fixture tests are still pending.
- Last verified checks during the port: `uv run poe rusttest`, `uv run poe rusttest_interop`, and `uv run poe rustdoc`.

## Completed Python API Parity

### Dataset Helpers

Rust has `load_iris`, cached MNIST loading, cached Fashion-MNIST loading through the MNIST-format reader, `dataset_to_device`, and `split_classification_dataset`.
Rust now has a typed high-level image loading layer for MNIST-compatible cached datasets, including subset selection, class filtering, shuffling, truncation, device movement, and explicit preprocessing.
The default preprocessing is unit-range because range-checked quantum feature maps expect values in `[0, 1]`.
Standardized MNIST remains available through the explicit MNIST mean/std option for classical and hybrid classifier workflows.
Fashion-MNIST intentionally has no standardized preprocessing constant because the Python notebooks use unit-range `ToTensor()` behavior and no current caller needs a second convention.
Rust image dataset loading is intentionally cache-only; dataset download remains a Python-side workflow.

## Deferred Design Work

### `MPSParameter`

`MPSParameter` is intentionally not implemented yet.
The current decision is that `MPS` is tensor-like, while a future `MPSParameter` should be parameter-like and constructed from `&nn::Path`.
The expected shape is that `MPSParameter` owns registered leaf tensors and exposes a method such as `as_mps() -> MPS` by shallow-cloning tensor handles.
This keeps differentiable forward computations separate from parameter registration and avoids putting `VarStore` or `Path` inside `MPS`.

Open design questions:

- Whether `MPSParameter` should own open and periodic MPS layouts through one type or typed variants.
- Whether `MPSParameter` should expose structural mutation methods at all, or only expose differentiable tensor-like views for forward passes.
- How to document safe use of `normalized`, orthogonalization, and projection when the source tensors are registered parameters.

## Missing Or Partial Python API Parity

### Generic Tensor Utilities

Rust ports the utility behavior needed by current algorithms, but not every Python helper exists as a public reusable function.
The `rust/einops-rs` submodule has `einsumstr!`, `einsum_str`, `contract_str!`, and `contract_str` helpers for readable named-dimension patterns.
The tensor-network Rust crate now uses `tensor_contract` for named shared-dimension contractions, `einsumstr!` for fixed literal production contractions, and `einops!` for fixed literal layout transforms in TEBD, MPS two-body reduced-density matrices, ResMPS boundary-vector expansion, and ADQCRNN auxiliary-state batching and flattening.
Dynamic-layout sites such as `gate_outer_product` and QRNN basis projection intentionally remain explicit because their layouts depend on runtime qubit counts and basis selection.

Remaining work:

- Consider typed wrappers for common shape assumptions such as state tensors, gate tensors, and feature-mapped samples if assertions become too scattered.

### Gate API Overloads

The important gate functions are present, but the Rust signatures are stricter than Python.
This is mostly desirable, but a few Python convenience overloads are not mirrored.

Remaining work:

- Decide whether to add a higher-level `rotate` API that accepts scalar parameters as well as a parameter vector, or keep only `rotate_from_params`.
- Add a direct nonbatched helper name only if it improves API clarity, because Rust `apply_gate` already covers the nonbatched path.
- Add parity tests for controlled gates, random gates, and gate outer products against Python fixtures.

## Algorithm And Workflow Gaps

### GMPS Workflows

The core GMPS training, NLL evaluation, selected-feature NLL evaluation, generation, and classification paths exist.
The Rust port intentionally omits Python-only runtime variants such as `torch.compile`, `vmap`, progress bars, and multiprocessing.

Remaining work:

- Add parity tests for `eval_nll_selected_features` and `gmps_classify_with_selected_features` against Python outputs.
- Add tests for `train_gmps` on a tiny deterministic dataset to catch sweep-direction and center-movement regressions.
- Decide whether Rust needs an explicit batch-index split helper, or whether the current inline batching is enough.
- Keep multiprocessing out of Rust unless a real use case appears.

### TEBD And Imaginary-Time Evolution

TEBD and dense imaginary-time evolution are implemented, but they are currently best treated as first-pass ports.

Remaining work:

- Add small Python-oracle tests for `evolve_gate_2body`, `calculate_mps_local_energies`, and `tebd`.
- Add small Python-oracle tests for dense `imaginary_time_evolution`.
- Review the two-body gate factorization in TEBD against the notebooks and reference implementation before relying on it for scientific results.
- Add CUDA tests for these paths, especially where SVD, QR, and matrix exponentials appear.

### Ground-State And Eigen Helpers

Rust has dense `calc_ground_state` and `eigs_power`.
The dense ground-state implementation is appropriate for notebook-scale systems, but it is not a sparse linear-operator replacement for SciPy `eigsh`.

Remaining work:

- Add Python-oracle tests comparing dense Rust `calc_ground_state` with Python `eigsh` on tiny Hamiltonians.
- Document size limits for dense `calc_ground_state`.
- Add tests for all four `EigenSelection` modes, not only largest and smallest algebraic.

### Tensor Decomposition

Rank-1 decomposition, gradient-based rank-1 decomposition, matrix unfolding, Tucker decomposition, and reduced matrices are present.
The current tests are small smoke tests, not full numerical parity tests.

Remaining work:

- Add Python-oracle tests for `rank1_tc`, `rank1_decomposition`, and `rank1_decomposition_gradient_based`.
- Review optimizer behavior in the gradient-based version, because Rust uses direct autograd updates rather than a full Adam optimizer.
- Add CUDA tests for decomposition paths that use SVD.

## Neural-Network Follow-Ups

The named network structs are in place, so the remaining neural-network work is mostly training and parity.

Remaining work:

- Add Python-oracle forward parity tests for `ADQCNet`, `FCADQCHybridClassifier`, `ADQCRNN`, `ResMPSSimple`, and `ADQCTimeEvolution`.
- Add tiny training smoke tests with `nn::VarStore` and `nn::Optimizer` for each trainable model where useful.
- Add CUDA forward smoke tests for the trainable networks.
- Add save/load examples for model parameters once the intended artifact format is chosen.
- Decide whether `ResMPSSimple` should remain a direct parameterized network or later share code with `MPSParameter`.

## Artifact Compatibility

MPS safetensors format compatibility is a stated requirement, but it needs stronger tests.

Remaining work:

- Add a Python-saves, Rust-loads test fixture for MPS safetensors.
- Add a Rust-saves, Python-loads test fixture using `safetensors.torch.load_file`.
- Test the exact scalar center convention where `center == -1` means no center.
- Confirm behavior for CPU, CUDA, float32, float64, and the expected complex tensor cases once complex safetensors support is confirmed in the local `tch-rs` fork.

## CUDA And Device Validation

The current Rust checks validate CPU and Python interop.
CUDA support is required, but CUDA-specific Rust parity tests are still incomplete.

Remaining work:

- Add CUDA-gated Rust tests for MPS orthogonalization, tensor decomposition, gate application, selected-feature GMPS evaluation, and trainable network forward passes.
- Add tests that ensure MPS-only linalg fallback sends MPS device work to CPU while CUDA stays on CUDA.
- Add tests that verify generated tensors follow the input tensor device in dataset helpers, gates, and network forwards.
- Keep MPS device behavior code-review validated unless a macOS/MPS runner is available.

## Documentation Work

Remaining work:

- Update Rust docs after `MPSParameter` design is chosen.
- Add examples for `VarStore` ownership with trainable networks.
- Add examples for Python/PyTorch tensor interop tests.
- Add a short artifact compatibility note once cross-language safetensors fixtures exist.

## Suggested Next Order

1. Add Python-oracle tests for the already-implemented Rust APIs, starting with MPS safetensors compatibility and selected-feature GMPS.
2. Add CUDA-gated tests for the highest-risk linalg and network paths.
3. Add Python-oracle tests for the high-level image dataset helpers and cached MNIST/Fashion-MNIST workflows.
4. Design and implement `MPSParameter`.
5. Add small training examples and save/load workflows for trainable networks.
