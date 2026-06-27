# Tensor Network Notes

This repo contains my notes and code related to the course [Tensor Network](https://space.bilibili.com/401005433/lists/864780?type=season).

## Environment Setup

This is a [`uv`](https://github.com/astral-sh/uv) project. Setting up the environment is easy:

1. If you have not got `uv` installed, follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it.
2. Run `git submodule update --init --recursive` to fetch the reference code and Rust `tch-rs` fork.
3. Run `uv sync` to create an environment and get dependencies, including development dependencies that you need to run the code in a notebook.

## Run Tools

We use `poe` to run tools. Available commands are:
* `lab`: run Jupyter Lab
* `sync`: sync the code in notebooks to the `tensor_network` package
* `format`: format the code in notebooks
* Checking: use ruff linter to check code 
    * `check_tensor_network`: check the exported code in `tensor_network` package
    * `check`: check code in `.`
    * `check_all`: check code in `.` and `tensor_network`
* Notebook smoke checks:
    * `notebooks_cpu_smoke`: run non-MLX notebooks in smoke mode on CPU and write reports under `target/notebook-smoke`
    * `notebooks_cuda_smoke`: run non-MLX notebooks in smoke mode on CUDA and write reports under `target/notebook-smoke`
* `precommit`: run pre-commit hooks

Sample usage:
```shell
# if your shell detects venvs automatically, you can run poe directly
poe lab
# if your shell does not detect venvs automatically, you can run uv run poe directly
uv run poe lab
```

Torch notebooks use `TN_TORCH_DEVICE` to choose the requested device and `TN_NOTEBOOK_SMOKE=1` to reduce long training loops for compatibility checks.
The automatic device order is CUDA, then MPS, then CPU.
MLX notebooks are intentionally excluded from the Torch smoke tasks, and locally saved MPS checkpoints under `datasets/mps` are treated as optional artifacts.

## Rust Experiments

The repository also contains a Cargo library crate named `tensor-network-code` in `rust/` for step-by-step Rust experiments.

Useful commands:
* `poe rusttest`: run the Rust tests with the pinned PyTorch/libtorch environment
* `poe rustdoc`: build the Rust API documentation, copy local image assets, and print the local serving path
* `cd rust && cargo doc --no-deps`: build the Rust API documentation
* `poe rusttest_interop`: run the optional PyTorch-to-`tch-rs` tensor interop experiment with the pinned `uv` Torch environment

When serving the generated Rust docs with `dufs`, serve `rust/target/doc` as the web root and open `/tensor_network_code/index.html`.
For example, run `dufs rust/target/doc -p 5000`, then open `http://HOST:5000/tensor_network_code/index.html`.

The initial rustdoc experiments live in standard Rust doc comments in `rust/src/lib.rs`.
Images use normal Markdown links like `![image](images/image.png)` and reference the shared `images/` directory through `rust/images`.
The local image assets are copied into the generated rustdoc directory by `poe rustdoc`.
Equations are rendered by MathJax through `rust/.cargo/config.toml` and `rust/docs/rustdoc-header.html`.
The Rust crate uses `tch-rs` and the pinned PyTorch/libtorch package installed by `uv`.
The Rust helper scripts configure `LIBTORCH_USE_PYTORCH=1`, `PYTHONPATH`, and `LD_LIBRARY_PATH` before invoking Cargo.
The PyTorch interop experiment is behind Cargo feature `python-interop` because it also needs PyO3 and an embeddable CPython 3.12 interpreter.
The Rust crate uses the patched `tch 0.25.1` fork pinned as the `rust/tch-rs` submodule so it can build against the repo's `torch==2.12.1` Python dependency.

## Trained MPS Checkpoints

See [the repo](https://huggingface.co/mapleL/mnist_mps) on Huggingface.

## Contribution

Contributions are very welcome. Please file an issue or PR if you have any questions or suggestions.

A few points to note:

* The code should primarily live in notebooks, not Python scripts. We use `nbdev` to export useful code from notebooks to `tensor_network` package for reusability.
* Run `pre-commit install`, or `uv run pre-commit install` if your shell doesn't autodetect venv

## Acknowledgements

* Big thanks to Prof. Ran for the course
* Thanks to Gemini 2.0 and Claude Sonnet for transcribing a lot of equations
