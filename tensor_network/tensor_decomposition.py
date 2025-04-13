# AUTOGENERATED! DO NOT EDIT! File to edit: ../1-8.ipynb.

# %% auto 0
__all__ = ['outer_product', 'rank1_tc', 'rank1_decomposition', 'rank1_decomposition_gradient_based', 'make_matrix',
           'tucker_decomposition', 'reduced_matrix']

# %% ../1-8.ipynb 0
import torch
from tqdm.auto import tqdm
from typing import List, Tuple
import einops

# %% ../1-8.ipynb 3
def outer_product(vectors: List[torch.Tensor]) -> torch.Tensor:
    """
    Computes the outer product of a list of vectors.
    Args:
        vectors (List[torch.Tensor]): A list of 1D tensors (vectors).
    Returns:
        torch.Tensor: The outer product of the input vectors.
    """
    assert isinstance(vectors, list), "Input must be a list of tensors"
    assert all(isinstance(v, torch.Tensor) for v in vectors), "All elements must be tensors"
    for i, v in enumerate(vectors):
        assert v.dim() == 1, f"Expected 1D tensor, got {v.dim()}D tensor at index {i}"
    num_vectors = len(vectors)
    assert num_vectors >= 2, "At least two vectors are required for outer product"
    vec_dim_names = [f"v{i}" for i in range(num_vectors)]
    einsum_exp = "{input_string} -> {output_string}".format(
        input_string=",".join(vec_dim_names),
        output_string=" ".join(vec_dim_names),
    )
    return einops.einsum(*vectors, einsum_exp)

# %% ../1-8.ipynb 6
def rank1_tc(x, v=None, it_time=10000, tol=1e-14):
    import torch as tc
    # From: https://github.com/ranshiju/Python-for-Tensor-Network-Tutorial/blob/4c89b0766159d3495122ec39339e7bd019f10fdf/Library/MathFun.py#L231
    # 初始化向量组v
    if v is None:
        v = list()
        for n in range(x.ndimension()):
            v.append(tc.randn(x.shape[n], device=x.device, dtype=x.dtype))

    # 归一化向量组v
    for n in range(x.ndimension()):
        v[n] /= v[n].norm()

    norm1 = 1
    err = tc.ones(x.ndimension(), device=x.device, dtype=tc.float64)
    err_norm = tc.ones(x.ndimension(), device=x.device, dtype=tc.float64)
    for t in tqdm(range(it_time)):
        for n in range(x.ndimension()):
            x1 = x.clone()
            for m in range(n):
                # conj here because we need to "cancel" out vectors by taking the dot product
                x1 = tc.tensordot(x1, v[m].conj(), [[0], [0]])
            for m in range(len(v) - 1, n, -1):
                x1 = tc.tensordot(x1, v[m].conj(), [[-1], [0]])
            norm = x1.norm()
            v1 = x1 / norm
            err[n] = (v[n] - v1).norm()
            err_norm[n] = (norm - norm1).norm()
            v[n] = v1
            norm1 = norm
        if err.sum() / x.ndimension() < tol and err_norm.sum() / x.ndimension() < tol:
            break
    return v, norm1


#|export
def rank1_decomposition(tensor: torch.Tensor,
                        num_iter: int = 10000,
                        stop_criterion: str = "zeta",
                        eps: float = 1e-14) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Decomposes a tensor into a list of rank-1 tensors using the rank-1 decomposition algorithm based on optimization. This is my implementation.

    Args:
        tensor (torch.Tensor): The input tensor to be decomposed.
        num_iter (int): The maximum number of iterations to perform.
        stop_criterion (str): The stopping criterion for the algorithm. Can be "zeta" or "norms".
        eps (float): The tolerance for convergence.
    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: A tuple containing the decomposed vectors and the zeta value.
    """
    assert stop_criterion in ["zeta", "norms"]
    device = tensor.device
    t_shape = tensor.shape
    k = len(t_shape)
    decomposed_vecs = [torch.randn(d, dtype=tensor.dtype, device=device) for d in t_shape]
    decomposed_vecs = [v / v.norm() for v in decomposed_vecs]
    zeta = 1.

    if stop_criterion == "zeta":
        for _ in tqdm(range(num_iter)):
            for idx in range(k):
                vs = decomposed_vecs[:idx] + decomposed_vecs[idx + 1:]
                vs = [v.conj() for v in vs]
                outer = outer_product(vs).unsqueeze(idx)
                sum_indices = list(range(k))
                sum_indices.pop(idx)
                vi = (tensor * outer).sum(tuple(sum_indices))
                vi /= vi.norm()
                decomposed_vecs[idx] = vi

            # calculate zeta
            vs = [v.conj() for v in decomposed_vecs]
            zeta_new = (tensor * outer_product(vs)).sum().real
            if (zeta_new - zeta).norm() < eps:
                break
            zeta = zeta_new
    else:
        # FIXME: seems to have bugs, adapted from the ref implementation, because the iteration takes a lot longer than my implementation - "zeta"
        v_norm_diffs = torch.ones(k, dtype=torch.float32, device=device)
        zeta_diffs = torch.ones(k, dtype=torch.float32, device=device)
        for _ in tqdm(range(num_iter)):
            for idx in range(k):
                # contraction
                vs = decomposed_vecs[:idx] + decomposed_vecs[idx + 1:]
                vs = [v.conj() for v in vs]
                outer = outer_product(vs).unsqueeze(idx)
                sum_indices = list(range(k))
                sum_indices.pop(idx)
                vi = (tensor * outer).sum(tuple(sum_indices))
                # calculate diffs
                vi_norm = vi.norm()
                vi /= vi_norm
                v_norm_diffs[idx] = (decomposed_vecs[idx] - vi).norm()
                zeta_diffs[idx] = (vi_norm - zeta).norm()
                decomposed_vecs[idx] = vi
                zeta = vi_norm

            if v_norm_diffs.sum() / k < eps and zeta_diffs.sum() / k < eps:
                break

    return decomposed_vecs, zeta


def rank1_decomposition_gradient_based(tensor: torch.Tensor, num_iter: int = 1000, eps: float = 1e-14) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Decomposes a tensor into a list of rank-1 tensors using gradient-based optimization. This is my implementation.

    Args:
        tensor (torch.Tensor): The input tensor to be decomposed.
        num_iter (int): The maximum number of iterations to perform.
        eps (float): The tolerance for convergence.
    Returns:
        Tuple[List[torch.Tensor], torch.Tensor]: A tuple containing the decomposed vectors and the zeta value.
    """
    t_shape = tensor.shape
    decomposed_vecs = [torch.randn(d, dtype=tensor.dtype) for d in t_shape]
    decomposed_vecs = [v / v.norm() for v in decomposed_vecs]
    for dv in decomposed_vecs:
        dv.requires_grad_(True)

    zeta = torch.ones(1, dtype=tensor.dtype)
    adam = torch.optim.Adam(decomposed_vecs, lr=0.1)
    for _ in tqdm(range(num_iter)):
        outer = outer_product(decomposed_vecs)
        loss = (tensor - outer).norm()
        loss.backward()
        adam.step()
        adam.zero_grad()
        with torch.no_grad():
            norms = [v.norm() for v in decomposed_vecs]
            zeta_new = torch.prod(torch.tensor(norms))
            if (zeta_new - zeta).norm() < eps:
                break
            else:
                zeta = zeta_new

    with torch.no_grad():
        decomposed_vecs = [v / v.norm() for v in decomposed_vecs]
        return decomposed_vecs, zeta

# %% ../1-8.ipynb 18
def make_matrix(tensor: torch.Tensor, left_index: int) -> torch.Tensor:
    """
    Converts a tensor into a matrix by moving the specified index to the front and reshaping the tensor.
    Args:
        tensor (torch.Tensor): The input tensor to be converted.
        left_index (int): The index to be moved to the front, as the row index.
    Returns:
        torch.Tensor: The reshaped matrix.
    """
    order = tensor.ndim
    assert order >= 2
    assert 0 <= left_index < order
    t = torch.movedim(tensor, left_index, 0)
    t = t.reshape(t.shape[0], -1)
    return t

#|export
def tucker_decomposition(tensor: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], List[int]]:
    """
    Decomposes a tensor into a core tensor and a list of matrices using Tucker decomposition.
    Args:
        tensor (torch.Tensor): The input tensor to be decomposed.
    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], List[int]]: A tuple containing the core tensor, a list of matrices, and a list of ranks.
    """
    order = tensor.ndim
    assert order >= 2
    matrices_U = []
    ranks = []
    for i in range(order):
        matrix = make_matrix(tensor, i)
        U, S, Vh = torch.linalg.svd(matrix)
        matrices_U.append(U)
        zeros = torch.zeros_like(S)
        # see https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html
        rtol = max(matrix.shape[0], matrix.shape[1]) * torch.finfo(matrix.dtype).eps
        rank = (~torch.isclose(S, zeros, atol=1e-10, rtol=rtol)).to(torch.int32).sum().item()
        ranks.append(rank)

    core_tensor = tensor
    for i in range(order):
        core_tensor = torch.tensordot(core_tensor, matrices_U[i].conj(), dims=[[0], [0]])

    return core_tensor, matrices_U, ranks

#|export
def reduced_matrix(core_tensor: torch.Tensor, n: int) -> torch.Tensor:
    """
    Computes the reduced matrix from the core tensor by multiplying it with its conjugate transpose.
    Args:
        core_tensor (torch.Tensor): The core tensor from Tucker decomposition.
        n (int): The index to specify which mode corresponding to the nth qubit to reduce.
    Returns:
        torch.Tensor: The resulting reduced matrix.
    """
    order = core_tensor.ndim
    assert order >= 2
    assert 0 <= n < order
    matrix = make_matrix(core_tensor, n)
    return matrix @ matrix.conj().t()
