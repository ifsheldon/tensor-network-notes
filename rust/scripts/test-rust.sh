#!/bin/sh
set -eu

. "$(dirname "$0")/libtorch-env.sh"

cargo test "$@"
