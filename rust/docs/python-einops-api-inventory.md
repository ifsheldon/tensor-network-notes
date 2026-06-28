# Python `einops` API Inventory

This inventory records the public Python `einops` APIs that matter for deciding how the Rust port should use or adapt `einops`-style pattern strings.

Sources checked on 2026-06-27:

- Local package introspection from `einops 0.8.2` in this repository's `uv` environment.
- Official API reference pages for [`asnumpy`](https://einops.rocks/api/asnumpy/), [`parse_shape`](https://einops.rocks/api/parse_shape/), [`rearrange`](https://einops.rocks/api/rearrange/), [`reduce`](https://einops.rocks/api/reduce/), [`repeat`](https://einops.rocks/api/repeat/), [`einsum`](https://einops.rocks/api/einsum/), and [`pack`/`unpack`](https://einops.rocks/api/pack_unpack/).
- Official introduction notes for the core API family at <https://einops.rocks/>.

## Scope

The Rust tensor-network port should avoid re-inventing a full `einops` implementation unless the forked Rust crate cannot cover the needed behavior.

The first high-confidence API target is `einops.einsum`, because its readable multi-letter axis names can be lowered to the compact one-character axis labels accepted by backend einsum implementations.

Other `einops` APIs often need operation plans rather than a single einsum string, because they can require reshape, permute, view, reduce, expand, concatenate, or split.

## Stable Top-Level APIs

Local `einops.__all__` in version `0.8.2` is `EinopsError`, `asnumpy`, `einsum`, `pack`, `parse_shape`, `rearrange`, `reduce`, `repeat`, and `unpack`.

| API | Python signature observed locally | Pattern form | Single backend einsum string? | Notes for the Rust port |
| --- | --- | --- | --- | --- |
| `rearrange` | `rearrange(tensor: Tensor \| list[Tensor], pattern: str, **axes_lengths) -> Tensor` | Arrow pattern such as `b c h w -> b h w c`, with composition like `(c h w)` and decomposition like `(h h2)`. | Only for the trivial subset that is pure axis reordering without composed axes, new axes, removed axes, list inputs, stack, or concatenate behavior. | Full support needs a shape-aware operation plan with reshape and permute steps. |
| `reduce` | `reduce(tensor: Tensor \| list[Tensor], pattern: str, reduction: str \| Callable, **axes_lengths) -> Tensor` | Arrow pattern where axes absent from the right side are reduced. | Only for the `sum` subset after any required reshape, because omitting labels in einsum sums over them. | `mean` needs a scale factor, `min`, `max`, `prod`, `any`, `all`, and callable reductions are not expressible as one ordinary einsum string. |
| `repeat` | `repeat(tensor: Tensor \| list[Tensor], pattern: str, **axes_lengths) -> Tensor` | Arrow pattern that may add axes, tile axes, repeat axes, and reorder axes. | Generally no. | Einsum cannot create arbitrary repeated values or new non-singleton output axes from just the input pattern string, so this needs an expand/repeat/reshape/permute operation plan. |
| `einsum` | `einsum(*tensors_and_pattern: Tensor \| str) -> Tensor` | Comma-separated input terms and an arrow output term, with the pattern passed last in Python. | Yes, this is the core target. | Lower readable axis names like `batch channel` to compact labels like `a b`, preserve tensor order, validate repeated and output labels, and handle ellipsis deliberately. |
| `pack` | `pack(tensors: Sequence[Tensor], pattern: str) -> tuple[Tensor, list[Shape]]` | Compact no-arrow pattern with exactly one `*`, such as `b * c`. | No. | This is generalized concatenate/stack behavior with packed-shape bookkeeping. |
| `unpack` | `unpack(tensor: Tensor, packed_shapes: list[Shape], pattern: str) -> list[Tensor]` | Same compact no-arrow pattern style as `pack`. | No. | This is split plus reshape behavior driven by packed-shape metadata. |
| `parse_shape` | `parse_shape(x: Tensor, pattern: str) -> dict` | Space-separated axis names with `_` used to skip an axis. | Not applicable. | This is useful as a parser and validation target, but it does not lower to tensor execution. |
| `asnumpy` | `asnumpy(tensor: Tensor) -> np.ndarray` | No pattern. | Not applicable. | This is backend conversion, so it is out of scope for a string-only transformation. |
| `EinopsError` | Exception type. | No pattern. | Not applicable. | Rust should use typed errors for fallible parsing and transformation. |

## APIs Used In This Repository

The exported Python library uses only `einsum`, `rearrange`, and `repeat` from top-level `einops`.

The MLX-only path also uses `einops.array_api.repeat`, but MLX is outside the current Rust port scope.

Not used in the exported Python code: `reduce`, `pack`, `unpack`, `parse_shape`, and `asnumpy`.

Notebook source uses the same surface: mostly `einsum`, some `rearrange`, and `repeat` in `4-4.ipynb`, plus `einops.array_api.repeat` in `4-4-mlx.ipynb`.

## Design Implications

`einops.einsum` should be the first behavior checked against the Rust `einops-rs` fork because it is the main API used by the tensor-network code and the most direct string-lowering problem.

`rearrange` and `repeat` are also required for non-MLX parity, but they should be treated as tensor transformation operation plans rather than as einsum-string rewrites.

`reduce`, `pack`, `unpack`, `parse_shape`, and `asnumpy` do not need to block the current Rust tensor-network port because they are not used by the exported non-MLX Python library.

The plan representation should preserve user-written axis names for diagnostics, even when a backend execution path uses compact generated labels.
