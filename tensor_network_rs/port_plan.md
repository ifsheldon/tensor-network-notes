# Rust Port Plan (tch-rs)

## Goals
- Port Python modules in `tensor_network/` to Rust under `tensor_network_rs/` with functional parity for core notebook paths.
- Validate each module with unit tests derived from notebooks; keep clippy clean and formatting consistent.

## Current Status (2025-09-14)
- Completed
  - `src/constants.rs` with PyTorch-like tolerances (`RTOL_DEFAULT=1e-5`, `ATOL_DEFAULT=1e-8`).
  - `src/utils/` including `allclose`, `set_seed`, dtype/device helpers, and `utils/einsum.rs` with `named_einsum`.
  - `eigen_decomposition.rs` basic power-iteration variant with tests.
  - `tensor_gates/functional.rs`: `kron/kron2`, `gate_outer_product`, Pauli X/Y/Z/ID (complex Y), `rotate_from_scalars`, `heisenberg`, controlled gates, `apply_gate` (einsum-based) + batched loop variant.
  - `quantum_state/functional.rs`: reduced density matrix, observations, onsite/bipartite entanglement, projections, bond energies.
  - `mps/{functional,modules}.rs`: construction, orthogonalization (QR/SVD), normalization, inner product, TT decomposition, projections, global tensor.
- Algorithms: `gmps.rs` (NLL, gradient, training loop), `time_evolving_block_decimation.rs`, `imaginary_time_evolution.rs` (basic), `tensor_decomposition.rs` (Tucker tools), `quantum_kernels.rs`, `lazy_classifier.rs`.
- Sanity tests across modules; crate builds and tests run locally.

- Remaining / In Progress
  - `gmps::eval_nll_selected_features` for partial subsets (matrix-env path) — implement and test.
  - Vectorize `apply_gate_batched(_with_vmap)` (replace loop with batched contraction/bmm if practical).
  - Save/load for Rust types (e.g., `safetensors` or custom serde) — design and implement.
  - Expand test coverage: GMPS training assertions, kernel metrics edge cases, TEBD scenarios, complex gates rigor.
  - Performance passes and allocation reductions on hot einsum paths.

## Coverage vs Python (high-level)
- Utils
  - Ported: dtype checks, inverse permutation, unify tensor dtypes, named einsum, complex builders, allclose, seed.
  - Missing: `utils/tensors.py` helpers (identity_tensor, zeros_state, normalize/rescale, generic outer_product), `checking.is_notebook()` (low priority).
- Tensor Gates
  - Ported: `apply_gate` (non-batched), controlled gates, Pauli X/Y/Z/ID (complex Y), `kron`, `gate_outer_product`, `spin_operator`, `identity_gate_tensor`, `rotate` (as `rotate_from_scalars`), `heisenberg`.
  - Missing: `apply_gate_nonbatched` wrapper (trivial), full `apply_gate_batched_with_vmap` (vectorized), `rotate(params_vec=...)` convenience API.
- Quantum State
  - Ported: reduced density matrix, expectations, onsite/bipartite entropies, projection, bond energies.
  - Gap: `observe_bond_energies` with single Hamiltonian input (Vec<Tensor> only right now) — add overload.
- MPS
  - Ported: generators, orthogonalization (QR/SVD), normalization, inner/outer contractions, TT decomposition, one/two-body RDMs, entropy utilities, projection helpers; `MPS` struct (+ selected methods).
  - Missing: save/load (safetensors); explicit `project_one_qubit` sugar; some in-place APIs mirroring `*_` Python methods (optional, Rust tends to return new values or mutate `&mut self`).
- Algorithms
  - Ported: GMPS core (NLL, gradient, train), kernels, lazy classifier, TEBD (minimal), imaginary time evolution (basic), Tucker helpers.
  - New: GMPS sampling (`generate_sample_with_gmps`) and classification (`gmps_classify`, `gmps_classify_with_selected_features`).
  - Improved: TEBD sweep with two-site gates; non-nearest interactions via MPO-style factorization (gl/gr) like Python’s evolve_gate_2body, avoiding SWAP; center movement, SVD split with `max_virtual_dim`, normalization, and local-energy tracking.
  - Missing: Multiprocessing variant for GMPS classification; additional Trotter scheduling options; `calc_ground_state_linear_operator` (SciPy-based in Python — skip for now).
  - Extra (optional): dynamic OEE analysis and entanglement-ordered sampling protocols.

## Order of Work (going forward)
1) Finish selective-feature NLL and tests.
2) Batch/vectorized gate application; profile vs loop implementation.
3) Persistence API (save/load) design + minimal implementation.
4) Broaden algorithms and kernel tests; add example-driven docs where helpful.
5) Optional: port selected network modules used in notebooks if needed.

## Validation
- Commands: `cargo check` → `cargo clippy --all-targets -- -D warnings` → `cargo fmt` → `cargo test`.
- Tests use small, deterministic cases; prefer fixed seeds. Use `RTOL_DEFAULT/ATOL_DEFAULT` for comparisons.

## Conventions
- Layout mirrors Python modules (e.g., `mps/functional.rs`). Public API names follow Python where sensible.
- Error handling: prefer `tch::Result<T>`; escalate via `anyhow` if needed for higher-level flows.

## Notes & Exclusions
- Ignore MLX (`*-mlx.ipynb`, `tensor_network/mlx`).
- Numerical parity may require tolerances; some tests can be slightly flaky — keep tests stable via seeds and small sizes.

## Discovered TODOs and Potential Bugs (from side-by-side scan)
- Eigen Decomposition
  - Bug: `eigs_power` uses `matrix_exp(H)` and `matrix_exp(H@H)` for LA/LM without the `TAU` factor used in Python. Fix to `matrix_exp(TAU * H)` and `matrix_exp(TAU * (H@H))` respectively; keep `TAU=1e-2`. Add unit tests comparing Rust to Python for small matrices and all four modes (LA/SA/LM/SM).
- Hamiltonians
  - Heisenberg: Python takes `.real` on the `Y⊗Y` term to ensure real-valued Hamiltonian. Our version sums complex terms directly. Align by taking real part or by constructing explicitly-real Y⊗Y; add tests to confirm hermiticity and real-valued output for real couplings.
- GMPS
  - `eval_nll_selected_features`: currently panics for partial subsets. Implement “matrix-env” path or batched vmap-equivalent; add parity test with Python for random subsets.
  - Missing: `generate_sample_with_gmps`, `gmps_classify`, and `gmps_classify_with_selected_features` (plus a non-multiprocessing path). These depend on feature mapping and subset NLL.
- Tensor Gates
  - `apply_gate_batched_with_vmap`: currently loops; replace with vectorized contraction/bmm or documented fallback; add tests on correctness and perf notes.
  - API Parity: Python has `rotate(params_vec|scalars, dtype/device)`; Rust exposes `rotate_from_scalars(f64,...)`. Provide a `rotate(params_vec)` ergonomic wrapper and dtype/device choices.
  - `kron` variadic vs slice: keep slice API but consider a convenience variadic wrapper for parity.
- Quantum State
  - `observe_bond_energies`: support single-H input form and mixed list forms as in Python; add tests.
- MPS
  - Two-body RDM and orthogonalization: add notebook-derived assertions (shapes, traces=1, hermitian) to catch subtle index order issues.
  - Save/Load: implement via `safetensors` (map `center` and tensors) to match Python’s `save_to_safetensors`/`load_from_safetensors`.
- Kernels / Classifier
  - Signature parity: Python `metric_matrix_neg_log_cos_sin` uses `calculation_method={deduplicate,no_deduplicate}`; Rust currently takes `dedup: bool`. Mirror the enum-like parameter names for clarity.
  - Add guardrails against NaNs for large `theta` (match Python checks and error message).
- TEBD / ITE
  - TEBD: Python performs gate splitting, center movement, orthogonalization, truncation, and local energy tracking. Rust implementation is a minimal placeholder. Plan: implement gate evolution and sweep loops, expose truncation/max rank, and compute local energies via two-body RDM.
  - Imaginary time evolution: bring over convergence/`tau`-halving logic and energy printouts (behind a verbosity flag); currently “basic”.
- Feature Mapping (for algorithms that need it)
  - Missing: `cossin_feature_map`, `feature_map_to_qubit_state`, `linear_mapping` — needed by GMPS sampling/classifiers and some notebooks. Provide lightweight Rust equivalents in a `feature_mapping` module.
- Misc
  - Add small utilities from `utils/tensors.py` when/if needed by notebooks (identity_tensor, zeros_state for complex, normalize/rescale, named contraction helper).

## API Parity Notes (Rust ↔ Python)
- Function names that intentionally differ:
  - `rotate_from_scalars` (Rust) ↔ `rotate(ita,beta,delta,gamma)` (Python). Plan: add `rotate(params_vec: [f64;4])` and `rotate(ita,..)` overload.
  - `metric_matrix_neg_log_cos_sin(samples, theta, dedup: bool)` (Rust) ↔ `metric_matrix_neg_log_cos_sin(samples, theta, calculation_method=...)` (Python). Plan: mirror Python parameter to avoid confusion.
- Input flexibility: accept both matrix and tensor forms for gates where Python does; ensure both code paths are tested.

## Test Parity Plan (derive from notebooks)
- 0-utils-* and 1-*:
  - `eigs_power` modes and accuracy vs PyTorch/NumPy for small dims.
- 2-5/2-6/2-8:
  - `apply_gate` correctness (1- and 2-qubit, with/without controls), `rotate`, Pauli Y algebra (`Y^2=I`), Heisenberg hermiticity/realness, reduced density matrices and observations.
- 3-*:
  - MPS orthogonalization round-trips (left→right→left), TT decomposition reconstruction error, one-/two-body RDM properties, entropy utilities.
- 4-*:
  - GMPS NLL (full and selected-feature), training decreases NLL on toy data, kernels and lazy classifier predictions on synthetic clusters.
- 5-*:
  - TEBD and ITE: evolution norms, basic energy descent on tiny chains; confirm truncation behavior when enabled.

## Work Items (prioritized checklist)
- [ ] Fix `eigs_power` TAU handling; add tests for LA/SA/LM/SM.
- [x] Implement `eval_nll_selected_features` (subset) and tests (degenerate=full).
- [x] Add `feature_mapping` module (`cossin_feature_map`, `feature_map_to_qubit_state`, `linear_mapping`).
- [x] GMPS sampling and classification (single-process).
- [ ] Vectorize `apply_gate_batched_with_vmap` or document fallback with benchmarks.
- [ ] Heisenberg: ensure real-valued output for real couplings (Y⊗Y handling) + tests.
- [x] TEBD sweep with orthogonalization/truncation + local energies (nearest-neighbor).
- [x] GMPS sampling and classifiers; optional multiprocessing is out-of-scope for now.
- [ ] Persistence API for `MPS` via `safetensors` (save/load center + tensors).
- [ ] Signature parity updates for kernels and `rotate` convenience wrapper.
