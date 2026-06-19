#!/bin/sh
set -eu

export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"

if [ -z "${PYO3_PYTHON:-}" ]; then
    if /usr/bin/python3.12 -c 'import ctypes' >/dev/null 2>&1; then
        PYO3_PYTHON="/usr/bin/python3.12"
    else
        PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
    fi
fi
export PYO3_PYTHON

UV_SITE_PACKAGES="$(uv run python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
export PYTHONHOME="${PYTHONHOME:-$("$PYO3_PYTHON" -c 'import sys; print(sys.base_prefix)')}"
export PYTHONPATH="${UV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

TORCH_LIB_DIR="$(uv run python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cargo test --features python-interop pytorch_tensor_interop -- --nocapture
