#!/usr/bin/env sh
set -eu

. "$(dirname "$0")/libtorch-env.sh"

cargo doc --no-deps

image_dir="target/doc/tensor_network_code/images"
mkdir -p "$image_dir"
cp images/mps_example.png "$image_dir/"
cp images/tensor_network_examples.png "$image_dir/"
cp images/qft_example.png "$image_dir/"

cat <<'EOF'

Rustdoc generated:
  target/doc/tensor_network_code/index.html

When serving with dufs from the repository root, serve rust/target/doc as the web root, then open:
  /tensor_network_code/index.html

For example:
  dufs rust/target/doc -p 5000
EOF
