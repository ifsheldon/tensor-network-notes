# Rust Port Plan (tch-rs)

## Goals
- Port Python modules in `tensor_network/` to Rust under `tensor_network_rs/` with functional parity for core paths used in notebooks.
- Validate each module with tests derived from corresponding notebooks.

## Order of Work
1) Utils (from `0-utils-*.ipynb`, `tensor_network/utils/*`): tensor helpers, mapping, checking, small data.
2) Linear algebra primitives: `eigen_decomposition.py`, `tensor_decomposition.py` (as needed by later steps).
3) Tensor gates: `tensor_gates/{functional,modules,hamiltonians}.py`.
4) Quantum state + MPS: `quantum_state/functional.py`, `mps/{functional,modules}.py`.
5) Networks: `networks/{res_mps,qrnn,fc,hybrid,adqc,time_evolution}.py` (prioritize ones used in notebooks).
6) Algorithms: `algorithms/{imaginary_time_evolution,time_evolving_block_decimation,gmps,lazy_classifier,quantum_kernels,calc_ground_state_linear_operator}.py`.
7) Notebook-only glue (optional): convenience runners if needed for parity.

## Validation per Module
- Add tests: start with shape/value checks using tiny tensors; fix seeds.
- Commands: `cargo check` → `cargo clippy --all-targets -- -D warnings` → `cargo fmt` → `cargo test`.
- Allclose: use `RTOL=1e-5`, `ATOL=1e-8` constants from `src/constants.rs` (match PyTorch defaults). Document this in code.
- Gradients: when learning/optimizing, create `VarStore`, attach tensors to the store, then call optimizers.

## Conventions
- File layout: mirror Python names where sensible (e.g., `mps/functional.rs`). Prefer modules over one giant file.
- Naming: snake_case for fns/vars, UpperCamelCase for types. Public API mirrors Python function names where possible.
- Error handling: return `tch::TchError` or `anyhow::Result` consistently within the crate.

## Notes & Exclusions
- Ignore MLX (`*-mlx.ipynb`, `tensor_network/mlx`).
- Some tests are numerically sensitive; allow tolerances and occasional flakiness.
- If PyTorch behavior is ambiguous, prefer explicit tch-rs ops even if more verbose.

## TODOs / Gaps (tracked)
- tensor_gates.functional.rotate: depends on complex exponentials and potentially missing ops — TODO.
- tensor_gates.functional.apply_gate_batched(_with_vmap): no vmap in tch-rs; implement via reshape+bmm later — TODO.
- tensor_gates.functional.gate_outer_product and einsum-heavy paths: einsum not bound in tch-rs — re-express via reshape+matmul or skip — TODO.
- pauli_operator("Y"): requires convenient complex tensor constructors — TODO. Current code panics with clear message.

Update: pauli Y, rotate_from_scalars, named_einsum, heisenberg Hamiltonian, quantum_state + MPS core are ported. Remaining algorithms (gmps, quantum_kernels, lazy_classifier, tensor_decomposition variants) are not yet ported; will add incrementally. Save/load via safetensors is currently unimplemented — TODO.

## Next Steps
- [ ] Create `src/constants.rs` with `ATOL`, `RTOL` and docs.
- [ ] Scaffold `src/utils/` with tensor helpers (device, dtype, seeding, allclose).
- [ ] Port `eigen_decomposition` with tests from notebooks `1-*` referencing eigensolver outputs.
- [ ] Port core MPS ops (functional) and minimal model to pass basic shape/value tests.
