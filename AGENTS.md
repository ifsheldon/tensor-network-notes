# Repository Guidelines

## Project Structure & Module Organization
- Notebooks: numbered lessons at repo root (e.g., `1-6.ipynb`, `4-4-mlx.ipynb`). Source of truth for code.
- Python package: `tensor_network/` (algorithms, mps, networks, utils, gates, `mlx/`). Generated artifacts like `_modidx.py` are nbdev outputs — do not edit by hand.
- Rust crate: `tensor_network_rs/` for experiments; unit tests live in `src/lib.rs`.
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
- Gradients: create a `VarStore` when parameters require grads (see `tensor_network_rs/src/lib.rs`).
- Numerics: mirror PyTorch defaults. Define `ATOL` `= 1e-8` and `RTOL` `= 1e-5` in `src/constants.rs` and use for `allclose`. Add a comment noting they match PyTorch.
- Tests: derive from notebooks; place in `tensor_network_rs/tests/` or `src/*` with `#[test]`. Some tests are flaky; that’s acceptable. Prefer fixed seeds and small tensors.

## Agent Notes
- Follow the notebook‑first workflow and this file across the repo. When updating Python code, edit notebooks → `poe sync` → lint. Avoid refactors that change lesson numbering or file naming without discussion.
