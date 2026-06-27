#!/bin/sh
set -eu

export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"

if [ -z "${PYO3_PYTHON:-}" ]; then
    if /usr/bin/python3.12 -c 'import ctypes' >/dev/null 2>&1; then
        PYO3_PYTHON="/usr/bin/python3.12"
    elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYO3_PYTHON="${VIRTUAL_ENV}/bin/python"
    elif [ -x "../.venv/bin/python" ]; then
        PYO3_PYTHON="$(cd .. && pwd)/.venv/bin/python"
    else
        PYO3_PYTHON="$(uv run python -c 'import sys; print(sys.executable)')"
    fi
fi
export PYO3_PYTHON
if [ "${PYO3_PYTHON}" = "/usr/bin/python3.12" ] && [ -z "${PYTHONHOME:-}" ]; then
    export PYTHONHOME="/usr"
fi

if [ -z "${VIRTUAL_ENV:-}" ] && [ -x "../.venv/bin/python" ]; then
    export VIRTUAL_ENV="$(cd ../.venv && pwd)"
    export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    ENV_PYTHON="${VIRTUAL_ENV}/bin/python"
elif [ -x "../.venv/bin/python" ]; then
    ENV_PYTHON="$(cd .. && pwd)/.venv/bin/python"
else
    ENV_PYTHON="${PYO3_PYTHON}"
fi

UV_SITE_PACKAGES="$("${ENV_PYTHON}" -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
export PYTHONPATH="${UV_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"

TORCH_LIB_DIR="$("${ENV_PYTHON}" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
