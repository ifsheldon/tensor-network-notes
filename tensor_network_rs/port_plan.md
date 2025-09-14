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
  - DONE: `eigs_power` uses `matrix_exp(tau*H)` / `matrix_exp(±tau*H^2)`; residual tests for LA/SA/LM/SM.
- Hamiltonians
  - DONE: Heisenberg uses real(Y⊗Y) to ensure real-valued output for real couplings; test added.
- GMPS
  - DONE: `eval_nll_selected_features` implemented (matrix-env path) + degeneracy test.
  - DONE: `generate_sample_with_gmps`, `gmps_classify`, and `gmps_classify_with_selected_features` (single-process).
- Tensor Gates
  - DONE: `apply_gate_batched_with_vmap` vectorized via named einsum over batch.
  - DONE: Added `rotate([ita,beta,delta,gamma], ...)` wrapper.
- Quantum State
  - DONE: `observe_bond_energies_single` supports single-H input alongside multi-H variant.
- MPS
  - DONE: Tests for two-body RDM hermiticity and unit trace post-normalization.
  - TODO: Save/Load via `safetensors` (map `center` and tensors).
- Kernels / Classifier
  - DONE: `metric_matrix_neg_log_cos_sin_method(samples, theta, calculation_method)` and NaN guardrails.
- TEBD / ITE
  - DONE: TEBD sweep implemented with MPO-style long-range gates (gl/gr), center movement, SVD truncation, and local energy tracking.
  - TODO: ITE convergence controls parity (tau-halving verbosity toggles).
- Feature Mapping
  - DONE: `cossin_feature_map`, `feature_map_to_qubit_state`, `linear_mapping` added.
- Misc
  - Optional: add `utils/tensors` helpers when needed (identity_tensor, zeros_state, normalize/rescale).

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
- [x] Fix `eigs_power` TAU handling; add tests for LA/SA/LM/SM.
- [x] Implement `eval_nll_selected_features` (subset) and tests (degenerate=full).
- [x] Add `feature_mapping` module (`cossin_feature_map`, `feature_map_to_qubit_state`, `linear_mapping`).
- [x] GMPS sampling and classification (single-process).
- [x] Vectorize `apply_gate_batched_with_vmap`.
- [x] Heisenberg: ensure real-valued output for real couplings + tests.
- [x] TEBD sweep with orthogonalization/truncation + local energies (incl. long-range via MPO gl/gr).
- [ ] Persistence API for `MPS` via `safetensors` (save/load center + tensors).
- [x] Signature parity updates for kernels and `rotate` convenience wrapper.
