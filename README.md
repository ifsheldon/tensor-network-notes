# Tensor Network Notes

This repo contains my notes and code related to the course [Tensor Network](https://space.bilibili.com/401005433/lists/864780?type=season).

## Environment Setup

This is a [`uv`](https://github.com/astral-sh/uv) project. Setting up the environment is easy:

1. If you have not got `uv` installed, follow the [instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it.
2. Run `uv sync` to create an environment and get dependencies, including development dependencies that you need to run the code in a notebook.

## Run Tools

We use `poe` to run tools. Available commands are:
* `lab`: run Jupyter Lab
* `sync`: sync the code in notebooks to the `tensor_network` package
* `format`: format the code in notebooks
* Checking: use ruff linter to check code 
    * `check_tensor_network`: check the exported code in `tensor_network` package
    * `check`: check code in `.`
    * `check_all`: check code in `.` and `tensor_network`
* `precommit`: run pre-commit hooks

Sample usage:
```shell
# if your shell detects venvs automatically, you can run poe directly
poe lab
# if your shell does not detect venvs automatically, you can run uv run poe directly
uv run poe lab
```

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
