#!/bin/sh
set -eu

. "$(dirname "$0")/libtorch-env.sh"

cargo test --features python-interop --test pytorch_interop -- --nocapture
