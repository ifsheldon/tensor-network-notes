# Repository Guidelines

## Project Structure & Module Organization
- Notebooks: numbered lessons at repo root (e.g., `1-6.ipynb`, `4-4-mlx.ipynb`). Source of truth for Python code.
- Python package: `tensor_network/` (algorithms, mps, networks, utils, gates, `mlx/`). Generated artifacts like `_modidx.py` are nbdev outputs — do not edit by hand.
- Rust crate: `tensor_network_rs/` (tch-rs port). Key modules under `src/`:
  - `constants.rs`, `utils/{mod,einsum,checking,mapping}.rs`
  - `tensor_gates/`, `quantum_state/`, `mps/{functional,modules}.rs`
  - `algorithms/{gmps,imaginary_time_evolution,time_evolving_block_decimation,tensor_decomposition,quantum_kernels,lazy_classifier}.rs`
  - Status/TODOs tracked in `tensor_network_rs/port_plan.md`.
- Assets: `images/`. Config: `pyproject.toml`, `.pre-commit-config.yaml`, `settings.ini`.

## Build, Test, and Development Commands
- Environment: `uv sync` (Python ≥3.12). Install hooks: `uv run pre-commit install`.
- Launch notebooks: `uv run poe lab`.
- Export notebook code → package: `uv run poe sync`.
- Format + lint: `uv run poe format` and `uv run poe check_all`.
- Python tests (if added): `uv run pytest`.
- Rust tests: `cd tensor_network_rs && cargo test` (build: `cargo build --release`).

## Coding Style & Naming Conventions
- Python formatting: `ruff format` (line length 100). 4‑space indent; snake_case for files/modules; UpperCamelCase for classes; lower_snake_case for functions/vars.
- Notebook naming: `N-M.ipynb`; MLX variants use `-mlx` suffix (e.g., `1-2-mlx.ipynb`).
- Do not modify generated files in `tensor_network/_modidx.py`; update notebooks and run `poe sync`.

## Testing Guidelines
- Prefer small, deterministic examples; fix random seeds where relevant.
- PyTest: place tests under `tests/` as `test_*.py`; assert tensor shapes/values and edge cases.
- Rust: add `#[test]` functions in `tensor_network_rs/src/*`; run `cargo test` locally and in CI (if configured).

## Commit & Pull Request Guidelines
- Commits: short, imperative, scoped (e.g., "restructure code", "improve ground state calc"). Group related changes; avoid noisy notebook outputs when possible.
- Before pushing: run `uv run poe precommit` (formats, syncs, lints).
- PRs: clear description of motivation and scope, linked issues, notes on notebooks touched, and any perf/accuracy impact. Include minimal repro snippets or screenshots where useful.

## Security & Configuration Tips
- Do not commit large datasets/checkpoints; use external storage (e.g., Hugging Face). Keep secrets out of notebooks and configs. Prefer relative imports and pinned deps defined in `pyproject.toml`.

## Rust Porting Workflow (tch-rs)
- Scope: port Python `tensor_network/` to Rust in `tensor_network_rs/`. Ignore MLX notebooks and `tensor_network/mlx`.
- Order: follow notebooks (e.g., `0-utils-*` → core ops → MPS → algorithms). Implement shared utils first if needed.
- Commands per module: `cargo check` → `cargo clippy --all-targets -- -D warnings` → `cargo fmt`.
- Numerics: mirror PyTorch defaults. Define and use `RTOL_DEFAULT=1e-5`, `ATOL_DEFAULT=1e-8` from `src/constants.rs`.
- Tests: derive from notebooks; place in `tensor_network_rs/tests/` or inline in `src/*` with `#[test]`. Prefer fixed seeds and small tensors.
- Einsum: prefer `utils::einsum::named_einsum` for readability when equations mirror the math; otherwise use `Tensor::einsum` or reshape+matmul.
- Complex: use `Kind::Complex{Float,Double}` when needed (e.g., Pauli Y, rotation gates). Helpers in `utils::mapping` and `utils::complex_from_slices` ease dtype conversions.
- Gradients: use `tch` autograd where viable; for GMPS-style training loops we currently compute explicit gradients as in the Python version.

Known TODOs captured in `tensor_network_rs/port_plan.md`:
- Selective-feature GMPS NLL (partial subset path).
- Vectorized batched gate application.
- Save/load for Rust models (e.g., safetensors/serde).

## Agent Notes
- Follow the notebook‑first workflow and this file across the repo. When updating Python code, edit notebooks → `poe sync` → lint. Avoid refactors that change lesson numbering or file naming without discussion.
