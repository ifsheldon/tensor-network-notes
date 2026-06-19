# Notebook Test Inventory

This inventory records implicit tests, assertions, reference comparisons, and smoke checks found in the notebooks. Cell numbers are 0-based. Use the `Promote` column as a first-pass backlog for turning notebook checks into explicit automated tests while keeping the notebooks as the source of truth.

## Reading Notes

- `Yes` means the notebook cell already contains a deterministic or nearly deterministic contract that should become a real test.
- `Maybe` means the cell is useful but needs seeding, smaller fixtures, explicit assertions, or artifact isolation before it belongs in the normal test suite.
- `No` means no implicit test was found, or the cell is only narrative, plotting, import boilerplate, or expensive interactive exploration.
- Reference dependencies name external comparison implementations, saved datasets, saved model artifacts, or runtime-specific dependencies that a future test must account for.

## Chapter 0: Utilities

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [0-utils-checking.ipynb](0-utils-checking.ipynb) | 4 | `is_notebook` | Asserts the current execution context is a notebook. | Notebook runtime | Maybe, but only as an environment-specific test. |
| [0-utils-data.ipynb](0-utils-data.ipynb) | 5, 6 | Dataset loading helpers | Smoke-loads Fashion-MNIST, MNIST, and `load_mnist_images`. | Torchvision cache and local dataset availability | Maybe, as integration tests outside the fast suite. |
| [0-utils-mapping.ipynb](0-utils-mapping.ipynb) | 3 | `unify_tensor_dtypes` | Smoke-runs dtype unification. | None | Yes, add explicit dtype and value assertions. |

## Chapter 1: Tensor Basics

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [1-2.ipynb](1-2.ipynb) | 2, 6, 10, 11 | Kronecker product expressions | Several cells print intermediate comparisons; cell 11 asserts `torch.kron`, `einsum`, and `einops.rearrange` produce the same tensor. | Torch and einops | Yes, especially cell 11. |
| [1-2-mlx.ipynb](1-2-mlx.ipynb) | 2, 6, 10, 11 | MLX Kronecker product expressions | Mirrors the Torch notebook; cell 11 asserts `mx.kron`, `einsum`, and rearrangement produce the same tensor. | MLX runtime | Yes if MLX is supported in CI. |
| [1-3.ipynb](1-3.ipynb) | 4, 8, 11, 12 | Tensor inner products, matrix multiplication, contraction ordering | Cell 4 asserts inner product equivalence and a negative ordering check; cell 8 asserts `matmul` and `einsum` agree; cell 11 prints contraction shape; cell 12 asserts staged contraction equals the direct `einsum` and includes a wrong-order negative check. | Torch | Yes, with an explicit shape assertion for cell 11. |
| [1-3-mlx.ipynb](1-3-mlx.ipynb) | 4, 8, 11, 12 | MLX tensor inner products, matrix multiplication, contraction ordering | Mirrors the Torch notebook with MLX operations and the same equivalence checks. | MLX runtime | Yes if MLX is supported in CI. |
| [1-4.ipynb](1-4.ipynb) | 4 | Identity tensor contraction | Asserts identity tensor contraction matches direct tensor multiplication behavior. | Torch | Yes, also cover exported `identity_tensor`. |
| [1-4-mlx.ipynb](1-4-mlx.ipynb) | 4 | MLX identity tensor contraction | Mirrors the Torch identity tensor equivalence check. | MLX runtime | Yes if MLX is supported in CI. |
| [1-5.ipynb](1-5.ipynb) | 2, 3 | Autograd basics | Prints gradients for simple expressions. | Torch autograd | Maybe, convert the printed gradient checks to explicit expected values. |
| [1-5-mlx.ipynb](1-5-mlx.ipynb) | 2, 3 | MLX autograd basics | Prints gradients for simple expressions. | MLX runtime | Maybe, convert the printed gradient checks to explicit expected values. |
| [1-6.ipynb](1-6.ipynb) | 3, 4, 8, 9, 10, 12, 16, 17 | Eigenvalue decomposition and power iteration | Cell 3 prints EVD reconstruction; cell 4 asserts eigenpair validity; cells 8-10 are plotting and optimization examples; cell 12 prints Torch, SciPy, and `LinearOperator` comparisons; cells 16-17 compare `eigs_power` against `Library.ExampleFun.eigs_power`. | SciPy and `Library.ExampleFun.eigs_power` | Yes for cells 4 and 17; maybe for cell 12 after adding numeric assertions and avoiding monkeypatching. |
| [1-6-mlx.ipynb](1-6-mlx.ipynb) | 3, 4, 8, 9, 10, 12, 15 | MLX eigenvalue decomposition and power iteration | Mirrors the Torch workflow; cell 4 asserts eigenpair validity; cell 15 compares MLX `eigs_power` against an inline Torch reference. | MLX runtime and Torch reference | Yes for cells 4 and 15 if MLX is supported in CI. |
| [1-7.ipynb](1-7.ipynb) | none | None | No implicit checks found. | None | No. |
| [1-8.ipynb](1-8.ipynb) | 4, 9, 11, 13, 15, 20, 21 | Outer products, rank-1 tensor completion, Tucker reconstruction, reduced matrices | Cell 4 checks `outer_product`; cells 9 and 11 compare rank-1 behavior but may be flaky; cell 13 compares with `rank1_tc`; cell 15 checks `rank1_tc` consistency; cell 20 checks Tucker reconstruction; cell 21 is print-only for `reduced_matrix`. | Internal `rank1_tc` and randomized inputs | Yes for cells 4, 15, and 20; maybe for rank-1 randomized checks after seeding. |

## Chapter 2: Quantum States and Hamiltonians

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [2-1.ipynb](2-1.ipynb) | none | None | No code or check cells found. | None | No. |
| [2-2.ipynb](2-2.ipynb) | none | None | No code or check cells found. | None | No. |
| [2-3.ipynb](2-3.ipynb) | none | None | No code or check cells found. | None | No. |
| [2-4.ipynb](2-4.ipynb) | none | None | No code or check cells found. | None | No. |
| [2-5.ipynb](2-5.ipynb) | 4, 7 | `apply_gate` and `TensorPureState` gate application | Cell 4 has input, shape, and index contract assertions; cell 7 compares matrix-form and tensor-form gate application against `TensorPureState.act_single_gate` for 1 to 4 qubit gates. | `Library.QuantumState.TensorPureState` | Yes. |
| [2-6.ipynb](2-6.ipynb) | 4, 7, 8 | `calc_reduced_density_matrix` and `calc_observation` | Cell 4 has qubit-index and operator contract assertions; cell 7 checks all subsystem combinations against the reference reduced density matrix; cell 8 checks observations against the reference implementation. | `Library.QuantumState.TensorPureState` | Yes. |
| [2-7.ipynb](2-7.ipynb) | none | None | No code or check cells found. | None | No. |
| [2-8.ipynb](2-8.ipynb) | 6, 8, 9, 10, 11 | `heisenberg`, `kron`, `gate_outer_product`, and ground-state energy calculations | Cell 6 asserts `kron` receives at least two matrices; cell 8 checks tensor Hamiltonian assembly against explicit Kronecker matrix assembly; cells 9-11 print matching ground-state energies from dense eigvalsh, SciPy `eigsh`, and `calc_ground_state`. | SciPy `eigsh` and internal `calc_ground_state` | Yes for cells 6 and 8; maybe for cells 9-11 after replacing prints with numeric assertions. |
| [2-8-calc-ground-state.ipynb](2-8-calc-ground-state.ipynb) | 2 | `calc_ground_state` input contracts | Asserts bounds for `smallest_k`, qubit count, Hamiltonian tensor shape, interaction positions, and unique qubits per interaction. | None | Yes for contract tests; add a small exact-diagonalization positive test. |

## Chapter 3: Differentiable Quantum Circuits

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [3-1.ipynb](3-1.ipynb) | 3, 4, 7, 8 | `pauli_operator`, `rotate`, `zeros_state`, and differentiability demo | Cells 3 and 7 contain contract assertions; cell 4 compares `rotate` against `Library.MathFun.rotate` for random parameter vectors; cell 8 is a long optimization smoke demo with printed decreasing loss. | `Library.MathFun.rotate` | Yes for cell 4; maybe for cell 8 after seeding, shortening, and asserting loss decrease. |
| [3-2.ipynb](3-2.ipynb) | 5 | Latent-gate SVD ADQC optimization | Long training smoke demo with printed loss decrease and no assertions. | None | Maybe, only as a short seeded smoke test. |
| [3-3.ipynb](3-3.ipynb) | none | None | Import and export boilerplate only. | None | No. |
| [3-4.ipynb](3-4.ipynb) | 2, 4, 7 | `QuantumGate`, `ParameterizedGate`, `SimpleGate`, `PauliGate`, `ADQCGate`, and `RotateGate` | Cell 2 has constructor and forward contract assertions; cell 4 smoke-runs Pauli, ADQC, and rotate gates on a 2-qubit state; cell 7 is a long optimization smoke demo. | None | Yes for cells 2 and 4 after adding explicit output assertions; maybe for cell 7 after shortening and seeding. |
| [3-5.ipynb](3-5.ipynb) | 3, 6, 12, 13, 16, 20, 21, 26, 28, 29, 30 | Batched gate application, `ADQCNet`, feature mapping, dataset splitting, classifier probabilities, and accuracy | Cell 3 has batched-gate contracts; cell 6 checks batched, nonbatched, and `vmap` gate paths against `TensorPureState`; cells 12, 16, 20, 21, 26, and 28 contain shape, range, and contract assertions; cell 13 prints ADQC output shape; cells 29-30 train an Iris classifier and print test accuracy. | `Library.QuantumState.TensorPureState` and Iris dataset loader | Yes for cell 6 and contract cells; maybe for training after replacing it with deterministic fixtures. |
| [3-6.ipynb](3-6.ipynb) | 4, 5, 10, 12, 16, 17, 18 | `feature_map_to_qubit_state`, `ADQCRNN`, and synthetic series utilities | Cells 4, 5, 10, and 12 contain shape and parameter contract assertions; cells 16-18 instantiate, train, and evaluate QRNN as a smoke workflow without explicit assertions. | None | Yes for contract cells; maybe for training workflow after shortening and seeding. |
| [3-7.ipynb](3-7.ipynb) | 3, 14, 15 | `FCADQCHybridClassifier` | Cell 3 asserts forward input shape; cells 14-15 train and evaluate an MNIST hybrid classifier and print final accuracy. | MNIST dataset loader | Yes for cell 3; no for the full training cells as-is. |
| [3-8.ipynb](3-8.ipynb) | 5, 6, 9, 11, 24 | `gate_outer_product`, `spin_operator`, `PolarizationGate`, and `ADQCTimeEvolution` | Cell 5 has arity contracts; cell 6 checks `gate_outer_product` against chained `torch.kron` for random 1 to 4 qubit factors in matrix and tensor forms; cells 9 and 11 have direction, shape, and time-step contracts; cell 24 is a long optimization smoke demo. | `torch.kron` and later exact-diagonalization context | Yes for cell 6 and contract cells; no for cell 24 as-is. |

## Chapter 4: Matrix Product States and Generative Tensor Networks

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [4-1.ipynb](4-1.ipynb) | 5, 14, 16, 20, 21, 25 | MPS global tensor construction, norm factors, normalization, and inner products | Checks global tensor contraction against dense `tensordot`, norm-factor products against dense norm, efficient norm factors against a naive version, normalized MPS norm against 1, and inner product against dense tensor inner product. | Internal dense and `tensordot` reference code | Yes; split timing and plotting in cell 16 away from correctness. |
| [4-2.ipynb](4-2.ipynb) | 6, 10, 14, 16, 18, 21, 22 | MPS central orthogonal form, normalization, center-tensor spectra, and one-body RDM | Cells 6, 10, 14, 16, 18, and 21 compare step/range orthogonalization, global tensor preservation, norm behavior, singular spectrum, and one-body RDM against reference or dense paths; cell 22 is a commented known-failing diagnostic. | `Library.MatrixProductState.MPS_basic`, `Library.QuantumState.TensorPureState`, and dense references | Yes except cell 22; isolate the reference transpose workaround in cell 21. |
| [4-3.ipynb](4-3.ipynb) | 3, 5 | Tensor-Train decomposition and virtual-dimension clipping | Cell 3 checks TT decomposition reconstructs random states and passes orthogonality checks; cell 5 checks clipped center-orthogonalization error matches clipped TT decomposition error. | None | Yes, with fixed seeds. |
| [4-4.ipynb](4-4.ipynb) | 4, 14, 15, 16 | Feed-forward MPS machine learning model | Cell 4 checks an `einsum` residual contraction against two loop formulations; cells 14-16 are parameter, training, and accuracy smoke outputs. | Fashion-MNIST dataset loader | Yes for cell 4; no for full training as-is. |
| [4-4-mlx.ipynb](4-4-mlx.ipynb) | 4, 14, 15 | MLX feed-forward MPS machine learning model | Cell 4 checks the MLX `einsum` residual contraction against loop formulations; cell 14 is training smoke output; cell 15 is commented. | MLX runtime | Yes for cell 4 if MLX is supported in CI; no for training as-is. |
| [4-5.ipynb](4-5.ipynb) | 4, 12, 13, 20, 21, 25, 30, 31 | Generative MPS gradients, NLL, training helpers, labels, and dataset training smoke | Cell 4 contains contract invariants in `calc_gradient`, `eval_nll`, and `train_gmps`; cell 25 has contract checks for `labels_to_binary` and `prepend_labels`; other listed cells are training and evaluation smoke outputs with printed NLL. | Dataset-dependent training paths | Yes for contract behavior; maybe for a tiny synthetic `eval_nll` or `train_gmps` smoke test. |
| [4-6.ipynb](4-6.ipynb) | 3, 4, 5, 12, 13, 14, 17, 18, 22, 24 | Generative tensor networks and quantum sampling | Cells 3-5 check projection and generation contracts, RDM shape, and probability bounds; later cells are visual generation smoke tests using reference weights or saved artifacts. | `Library.BasicFun` and saved MPS artifacts | Yes for small synthetic projection and generation tests; no for visual artifact cells as-is. |
| [4-7.ipynb](4-7.ipynb) | 9, 10 | Generative MPS probability classifier | Cell 9 guards `gmps_classify` inputs; cell 10 computes classification accuracy as smoke output. | Saved `datasets/mps/mnist_*_mps.safetensors` artifacts | Yes for contract tests; maybe for a tiny known-label classifier fixture. |
| [4-8.ipynb](4-8.ipynb) | 6, 7, 16, 17, 18, 21, 23 | Quantum kernels, lazy states, distance functions, and classifier smoke checks | Cells 7 and 17 compare kernel and distance functions against reference code with `allclose`; cells 6, 16, and 18 contain shape, range, and NaN guards; cells 21 and 23 are classifier accuracy smoke computations. | `Library.MathFun`, MNIST data, and saved GMPS artifacts | Yes for cells 7 and 17 and for contract guards; maybe for classifier cells using synthetic fixtures. |
| [4-8-lazy-classifier.ipynb](4-8-lazy-classifier.ipynb) | none | None | Export and import setup only. | None | No. |
| [4-9.ipynb](4-9.ipynb) | 3, 18, 24, 31, 32 | Entanglement-based feature selection and selected-feature NLL | Cell 3 checks bipartite reduced-density entropies are close and prints a sampling comparison; cells 31 and 32 compare selected-feature NLL against full `eval_nll` and reference `generative_MPS`, but both are marked flaky; cells 18 and 24 are accuracy smoke outputs. | `Library.MatrixProductState.generative_MPS`, saved GMPS artifacts, and MNIST data | Yes for cell 3; maybe for cells 31 and 32 after stabilizing seeds, tolerances, and artifact setup. |
| [4-10.ipynb](4-10.ipynb) | 4, 5, 7, 9, 13 | Quantum measurement and dynamic feature selection | Cells 4, 5, and 9 guard input and shape contracts; cell 7 prints a dynamic-OEE demo on simple samples; cell 13 is a long timing smoke run for OEE variation. | Saved MPS artifacts and MNIST data for the long run | Yes for contract guards; maybe convert cell 7 to explicit numeric or shape assertions. |
| [4-11.ipynb](4-11.ipynb) | 3, 6, 8 | Tensor-network compressed sampling | Cell 3 guards selected-feature count; cell 6 is artifact-backed sampling smoke output; cell 8 is visualization only. | Saved MPS artifact | Maybe, add a synthetic test for output length, uniqueness, and index range. |

## Chapter 5: Ground-State Algorithms

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [5-1.ipynb](5-1.ipynb) | 4, 7, 9, 10 | `imaginary_time_evolution` contracts and exact ground-state comparison | Cell 4 asserts iteration, `tau`, Hamiltonian tensor form, init-state device rules, and `interaction_positions` type; cells 7, 9, and 10 print exact linear-operator energy near `-1.6160253286` and imaginary-time energy near `-1.6160255671`. | Internal `calc_ground_state` | Yes for contract tests; maybe for energy comparison after seeding and using a tolerance. |
| [5-2.ipynb](5-2.ipynb) | 4, 6, 9, 11, 13, 14, 15, 16, 17, 21, 23, 25, 29, 32, 34, 36, 38, 39 | TEBD, MPS two-body gates, RDMs, local energies, entropy, and imaginary-time convergence | Cell 4 checks identity tensor-product construction behind a disabled benchmark flag; cell 6 reconstructs a two-body gate from split factors; cell 9 guards `evolve_gate_2body`; cell 11 is a disabled reference comparison for `evolve_gate_2body`; cells 13-14 check two-body RDMs against partial trace, dense RDM, and `MPS_tebd.two_body_RDM`; cells 15-16 cover local-energy contracts and disabled reference comparison; cells 17, 23, and 25 cover TEBD contracts and observation comparisons; cell 29 guards observation helpers; cells 32, 34, 36, 38, and 39 print spectrum, bond energy, entropy, and convergence diagnostics. | `reference_code/Library.MatrixProductState.MPS_tebd` and internal `calc_ground_state` | Yes for cells 6, 9, 13, 14, 17, 23, 25, and 29; maybe for disabled or expensive cells after reducing fixtures and seeding. |

## Miscellaneous Notebooks

| Notebook | Cells | Tested behavior | Existing check | Reference dependency | Promote |
|---|---:|---|---|---|---|
| [import_reference.ipynb](import_reference.ipynb) | 0 | Reference-code import setup | Smoke behavior appends `reference_code` to `sys.path` and prints when added. | Local `reference_code` directory | Maybe, as an importability test for expected reference modules. |
| [tensor_gate_extra.ipynb](tensor_gate_extra.ipynb) | 2, 5, 6 | Fast identity-gate and controlled-gate tensor helpers | Cell 2 checks fast identity-gate tensor construction against a slow explicit-product identity; cell 5 runs random trials comparing fast controlled gates, slow tensor paths, and matrix-form paths via `view_gate_matrix_as_tensor`; cell 6 is a plot-only controlled-gate matrix pattern cell. | None | Yes for cells 2 and 5 with seeding and parameterized dtype or qubit counts; no for cell 6 as-is. |
| [tensor_product_experiments.ipynb](tensor_product_experiments.ipynb) | 4, 5 | `tensor_contract` and general tensor product contraction | Cell 4 guards arity, dimension spec type, string dimension names, and disjoint dimension sets; cell 5 asserts manual `BA` contraction equals direct `ba`, then asserts `tensor_contract` equals `ba`. | None | Yes, with a seeded random tensor fixture and shape assertion. |

## Suggested Promotion Order

1. Start with pure deterministic reference-equivalence checks that do not need datasets: `1-2`, `1-3`, `1-4`, `2-5`, `2-6`, `2-8`, `3-1`, `3-5`, `3-8`, `4-1`, `4-2`, `4-3`, `4-4`, `4-8`, `5-2`, `tensor_gate_extra`, and `tensor_product_experiments`.
2. Add input-contract tests next, especially cells that currently use inline `assert` statements inside exported definitions.
3. Keep long training, plotting, benchmark, dataset, and saved-artifact cells as demos unless they are reduced to seeded synthetic fixtures with explicit pass/fail criteria.
4. Treat MLX notebooks as a separate optional test target because they need a runtime dependency and possibly separate CI capability.
