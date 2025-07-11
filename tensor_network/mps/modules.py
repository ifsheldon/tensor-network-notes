"""在虚拟指标中间插入一个 Unitary 和它的 Hermitian 不会改变全局张量"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../4-2.ipynb.

# %% auto 0
__all__ = ['MPS']

# %% ../../4-2.ipynb 2
import torch
from typing import List, Tuple, Literal, Self
from .functional import gen_random_mps_tensors, MPSType

# %% ../../4-2.ipynb 12
from tensor_network.mps.functional import (
    orthogonalize_arange,
    calc_global_tensor_by_tensordot,
    calculate_mps_norm_factors,
    calc_inner_product,
    tt_decomposition,
)
import sys
from einops import einsum
from safetensors.torch import save_file, load_file


class MPS:
    """
    Matrix Product State (MPS) class.
    """

    def __init__(
        self,
        *,
        mps_tensors: List[torch.Tensor] | None = None,
        length: int | None = None,
        physical_dim: int | None = None,
        virtual_dim: int | None = None,
        mps_type: MPSType | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        requires_grad: bool | None = None,
    ) -> None:
        """
        Initialize an MPS. It can be initialized from a list of local tensors, or from paramters.

        Args:
            mps_tensors: List[torch.Tensor] | None, the local tensors of the MPS. If None, the MPS will be initialized from parameters.
            length: int | None, the length of the MPS. Should be provided if mps_tensors is None.
            physical_dim: int | None, the physical dimension of the MPS. Should be provided if mps_tensors is None.
            virtual_dim: int | None, the virtual dimension of the MPS. Should be provided if mps_tensors is None.
            mps_type: MPSType | None, the type of the MPS. Should be provided if mps_tensors is None.
            dtype: torch.dtype | None, the dtype of the MPS. Should be provided if mps_tensors is None.
            device: torch.device | None, the device of the MPS. Should be provided if mps_tensors is None.
            requires_grad: bool | None, whether the MPS requires gradient.
        """
        if mps_tensors is None:
            assert (
                length is not None
                and physical_dim is not None
                and virtual_dim is not None
                and mps_type is not None
                and dtype is not None
                and device is not None
                and requires_grad is not None
            ), (
                f"mps_tensors is None, so all arguments must be provided, but got {mps_tensors=}, {length=}, {physical_dim=}, {virtual_dim=}, {mps_type=}, {dtype=}, {device=}, {requires_grad=}"
            )
            mps_tensors = gen_random_mps_tensors(
                length, physical_dim, virtual_dim, mps_type, dtype, device
            )
            for i in range(len(mps_tensors)):
                mps_tensors[i].requires_grad = requires_grad
            self._length: int = length
            self._physical_dim: int = physical_dim
            self._virtual_dim: int = virtual_dim
            self._mps_type: MPSType = mps_type
            self._dtype: torch.dtype = dtype
            self._device: torch.device = device
        else:
            # TODO: checking whether the mps_tensors is valid, not emergent
            self._length: int = len(mps_tensors)
            self._physical_dim: int = mps_tensors[0].shape[1]
            self._virtual_dim: int = mps_tensors[0].shape[2]
            self._mps_type: MPSType = (
                MPSType.Open if mps_tensors[0].shape[0] == 1 else MPSType.Periodic
            )
            self._dtype: torch.dtype = mps_tensors[0].dtype
            self._device: torch.device = mps_tensors[0].device
            requires_grad = mps_tensors[0].requires_grad if requires_grad is None else requires_grad

        self._requires_grad: bool = requires_grad
        self._mps: List[torch.Tensor] = mps_tensors
        self.set_requires_grad_(requires_grad)
        self._center: int | None = None

    def set_requires_grad_(self, requires_grad: bool):
        """
        Set the requires_grad attribute of the MPS.

        Args:
            requires_grad: bool, whether the MPS requires gradient.
        """
        self._requires_grad = requires_grad
        for t in self._mps:
            t.requires_grad = requires_grad

    def save_to_safetensors(self, path: str):
        """
        Save the MPS to a safetensors file.

        Args:
            path: str, the path to save the MPS.
        """
        tensor_dict = {f"{i}": t for i, t in enumerate(self._mps)}
        tensor_dict["center"] = (
            torch.tensor(-1) if self.center is None else torch.tensor(self.center)
        )
        save_file(tensor_dict, path)

    @staticmethod
    def load_from_safetensors(path: str, requires_grad: bool) -> Self:
        """
        Load the MPS from a safetensors file.

        Args:
            path: str, the path to load the MPS.
            requires_grad: bool, whether the MPS requires gradient.

        Returns:
            MPS, the MPS loaded from the safetensors file.
        """
        tensor_dict = load_file(path)
        center = tensor_dict.pop("center").item()
        mps_tensors = [None] * len(tensor_dict)
        for i, t in tensor_dict.items():
            i = int(i)
            mps_tensors[i] = t
        mps = MPS(mps_tensors=mps_tensors, requires_grad=requires_grad)
        mps._center = None if center == -1 else center
        return mps

    def center_orthogonalization_(
        self,
        center: int,
        mode: Literal["svd", "qr"],
        truncate_dim: int | None = None,
        check_nan: bool = True,
        normalize: bool = False,
    ):
        """
        Perform center orthogonalization on the MPS. This is an in-place operation.

        Args:
            center: int, the center of the MPS.
            mode: Literal["svd", "qr"], the mode of orthogonalization.
            truncate_dim: int | None, the dimension to be truncated. If None, no truncation will be performed.
            check_nan: bool, whether to check the nan value in the results. If True, the nan value will be checked.
        """
        assert -self.length <= center < self.length, "center out of range"
        if center < 0:
            center = self.length + center
        if self._center is None:
            new_local_tensors = orthogonalize_arange(
                self._mps,
                0,
                center,
                mode,
                truncate_dim=truncate_dim,
                normalize=normalize,
                check_nan=check_nan,
            )
            new_local_tensors = orthogonalize_arange(
                new_local_tensors,
                self.length - 1,
                center,
                mode,
                truncate_dim=truncate_dim,
                normalize=normalize,
                check_nan=check_nan,
            )
            for i in range(self.length):
                self._mps[i] = new_local_tensors[i]
        elif self.center != center:
            new_local_tensors, changed_indices = orthogonalize_arange(
                self._mps,
                self.center,
                center,
                mode,
                truncate_dim,
                return_changed=True,
                check_nan=check_nan,
            )
            for changed_idx in changed_indices:
                self._mps[changed_idx] = new_local_tensors[changed_idx]
        else:
            # when self.center == center
            pass
        self._center = center
        if normalize:
            self.center_normalize_()

    def center_normalize_(self):
        """
        Normalize the center tensor of the MPS.
        """
        assert self.center is not None, (
            "The MPS is not center orthogonalized. Perform center orthogonalization first."
        )
        self._mps[self.center] /= self._mps[self.center].norm()

    def force_set_local_tensor_(self, i: int, value: torch.Tensor):
        """
        Force set the local tensor at index i to the given value with checking the shape and dtype.

        Args:
            i: int, the index of the local tensor to be set.
            value: torch.Tensor, the value to be set.
        """
        value = value.to(dtype=self._dtype, device=self._device)
        value.requires_grad = self._requires_grad
        self._mps[i] = value

    def __getitem__(self, i: int) -> torch.Tensor:
        return self._mps[i]

    def __setitem__(self, i: int, value: torch.Tensor):
        local_tensor_shape = self[i].shape
        local_tensor_dtype = self[i].dtype
        assert value.shape == local_tensor_shape, (
            f"value shape must match local tensor shape {local_tensor_shape}, but got {value.shape}"
        )
        assert value.dtype == local_tensor_dtype, (
            f"value dtype must match local tensor dtype {local_tensor_dtype}, but got {value.dtype}"
        )
        self.force_set_local_tensor_(i, value)

    def global_tensor(self) -> torch.Tensor:
        """
        Calculate the global tensor of the MPS.
        """
        # use tensordot to contract the mps tensors, because it's faster than calc_global_tensor_by_contract
        if self.length > 15:
            print(
                "Warning: Calculating global tensor of MPS with length > 15, this may up all the memory",
                file=sys.stderr,
            )
        return calc_global_tensor_by_tensordot(self._mps)

    def norm_factors(self) -> torch.Tensor:
        """
        Calculate the norm factors of the MPS.
        """
        return calculate_mps_norm_factors(self._mps).real

    def norm(self, *, _efficient_mode: bool = True) -> torch.Tensor:
        """
        Calculate the norm of the MPS.
        """
        if _efficient_mode and self.center is not None:
            return self._mps[self.center].norm()
        else:
            norm_factors = self.norm_factors()
            # use sqrt inside the product to avoid overflow
            return torch.prod(norm_factors.sqrt())

    def normalize_(self, *, _efficient_mode: bool = True):
        """
        Normalize the MPS in-place.
        """
        if _efficient_mode and self.center is not None:
            self._mps[self.center] /= self.norm()
        else:
            norm_factors = 1 / self.norm_factors().sqrt()
            for i in range(self.length):
                self._mps[i] *= norm_factors[i]

    def inner_product(self, other: "MPS", return_product_factors: bool = False) -> torch.Tensor:
        """
        Calculate the inner product of two MPS. These two MPS must have the same length.
        """
        assert isinstance(other, MPS), "other must be a MPS"
        assert self.length == other.length, "length of two MPS must be the same"
        product_factors = calc_inner_product(self._mps, other._mps)
        if return_product_factors:
            return product_factors
        else:
            return torch.prod(product_factors)

    def check_orthogonality(
        self, *, check_mode: Literal["print", "assert"] = "print", tolerance: float = 1e-6
    ):
        """
        Check the orthogonality of the MPS.
        """
        assert check_mode.lower() in ["print", "assert"], (
            "check_mode must be either 'print' or 'assert'"
        )
        print_mode = check_mode.lower() == "print"
        if self.center is None:
            print("center is None, so no orthogonality check can be performed")
        else:
            identity = torch.eye(
                2, dtype=self._dtype, device=self._device
            )  # cache for the identity matrix
            for i in range(self.length):
                if i == self.center:
                    if print_mode:
                        print(f"Local Tensor {i}: Center")
                else:
                    local_tensor = self._mps[i]
                    if i < self.center:
                        product = torch.einsum("abc,abd->cd", local_tensor.conj(), local_tensor)
                    else:
                        product = torch.einsum("xab,yab->xy", local_tensor, local_tensor.conj())

                    assert product.shape[0] == product.shape[1]

                    if identity.shape[0] != product.shape[0]:
                        identity = torch.eye(
                            product.shape[0], dtype=self._dtype, device=self._device
                        )

                    diff_norm = (product - identity).norm(p=1).item()

                    if print_mode:
                        print(f"Local Tensor {i}: {diff_norm}")
                    else:
                        assert diff_norm < tolerance, (
                            f"Local Tensor {i} is not orthogonal, {diff_norm=}"
                        )

    def to_(self, dtype: torch.dtype | None = None, device: torch.device | None = None) -> Self:
        """
        Convert the MPS to the given dtype and device in-place.

        Args:
            dtype: torch.dtype | None, the dtype to convert to.
            device: torch.device | None, the device to convert to.

        Returns:
            MPS, the MPS converted to the given dtype and device.
        """
        if dtype is not None and self._dtype != dtype:
            for i in range(self.length):
                self._mps[i] = self._mps[i].to(dtype=dtype)
            self._dtype = dtype
        if device is not None and self._device != device:
            for i in range(self.length):
                self._mps[i] = self._mps[i].to(device=device)
            self._device = device
        return self

    def one_body_reduced_density_matrix(
        self, *, idx: int, do_tracing: bool, inplace_mutation: bool = False
    ) -> torch.Tensor:
        """
        Calculate the one-body reduced density matrix of the MPS.

        Args:
            idx: int, the index of the qubit to calculate the reduced density matrix of.
            do_tracing: bool, whether to do tracing.
            inplace_mutation: bool, whether to do in-place mutation. Speed is faster if True.
        """
        assert 0 <= idx < self.length, "idx must be in [0, length - 1]"
        if self.center is None:  # TODO: optimize this branch
            # maybe we can just use einsum here, need some benchmarking
            if inplace_mutation:
                self.center_orthogonalization_(idx, "qr")
                return self.one_body_reduced_density_matrix(idx=idx, do_tracing=do_tracing)
            else:
                # do center orthogonalization out of place
                local_tensors = self.local_tensors
                length = len(local_tensors)
                center = idx
                mode = "qr"
                local_tensors = orthogonalize_arange(local_tensors, 0, center, mode)
                local_tensors = orthogonalize_arange(local_tensors, length - 1, center, mode)
                center_tensor = local_tensors[center]
        else:
            if self.center == idx:
                center_tensor = self.center_tensor
            else:  # TODO: optimize this branch
                if inplace_mutation:
                    self.center_orthogonalization_(idx, "qr")
                    return self.one_body_reduced_density_matrix(idx=idx, do_tracing=do_tracing)
                else:
                    # moving center out of place
                    local_tensors = self.local_tensors
                    new_center = idx
                    local_tensors = orthogonalize_arange(
                        local_tensors, self.center, new_center, mode="qr"
                    )
                    center_tensor = local_tensors[new_center]

        rdm = einsum(
            center_tensor,
            center_tensor.conj(),
            "left mid right, left mid_conj right -> mid mid_conj",
        )
        if do_tracing:
            return rdm / rdm.trace()
        else:
            return rdm

    @property
    def center_tensor(self) -> torch.Tensor | None:
        if self.center is None:
            return None
        else:
            return self._mps[self.center]

    @property
    def local_tensors(self) -> List[torch.Tensor]:
        return [i for i in self._mps]

    @property
    def length(self) -> int:
        return self._length

    @property
    def physical_dim(self) -> int:
        return self._physical_dim

    @property
    def virtual_dim(self) -> int:
        return self._virtual_dim

    @property
    def mps_type(self) -> MPSType:
        return self._mps_type

    @property
    def center(self) -> int | None:
        return self._center

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @staticmethod
    def from_state_tensor(
        state_tensor: torch.Tensor, max_rank: int | None = None, use_svd: bool = False
    ) -> Self:
        """
        Initialize an MPS from a state tensor.

        Args:
            state_tensor: torch.Tensor, the state tensor to initialize the MPS from.
            max_rank: int | None, the maximum rank of the MPS. If specified, the MPS will be truncated to the given rank using TT decomposition.
            use_svd: bool, whether to use SVD to truncate the MPS.

        Returns:
            MPS, the MPS initialized from the state tensor.
        """
        local_tensors, _ = tt_decomposition(state_tensor, max_rank=max_rank, use_svd=use_svd)
        mps = MPS(mps_tensors=local_tensors)
        mps._center = len(local_tensors) - 1
        return mps

# %% ../../4-6.ipynb 4
from .functional import project_multi_qubits as project_multi_qubits_func
from fastcore.basics import patch


@patch
def project_multi_qubits(
    self: MPS, qubit_indices: List[int], project_to_states: torch.Tensor | List[int]
) -> Self:
    """
    Do projection of multiple qubits of this MPS, returning a new MPS.

    Args:
        qubit_indices: List of indices of the qubits to project.
        project_to_states: List of states to project to.

    Returns:
        MPS, the new MPS after projection.
    """
    local_tensors = self._mps
    new_local_tensors = project_multi_qubits_func(local_tensors, qubit_indices, project_to_states)
    return MPS(mps_tensors=new_local_tensors)


@patch
def project_one_qubit(self: MPS, qubit_idx: int, project_to_state: torch.Tensor | int) -> Self:
    """
    Project one qubit of this MPS, returning a new MPS.

    Args:
        qubit_idx: int, the index of the qubit to project.
        project_to_state: torch.Tensor | int, the state to project to.
    """
    assert isinstance(project_to_state, (torch.Tensor, int)), (
        "project_to_state must be a tensor or an integer"
    )
    if isinstance(project_to_state, int):
        project_to_states = [project_to_state]
    else:
        assert project_to_state.ndimension() == 1, "project_to_state must be a 1D tensor"
        project_to_states = project_to_state.unsqueeze(0)

    qubit_indices = [qubit_idx]
    return self.project_multi_qubits(qubit_indices, project_to_states)

# %% ../../4-9.ipynb 9
@patch
def entanglement_entropy_onsite_(
    self: MPS, indices: List[int] | None = None, eps: float = 1e-10
) -> torch.Tensor:
    """
    Calculate the onsite entanglement entropies for qubits at `indices`. With inplace mutation.

    Args:
        indices: The indices of the qubits to calculate the entanglement entropies. If `None`, calculate for all qubits.
        eps: The small value to avoid log(0). Default is 1e-10.

    Returns:
        The entanglement entropies for the qubits at `indices`.
    """
    if indices is None:
        indices = list(range(self._length))
    assert 0 < len(indices) <= self._length, "indices must be a list of indices in [0, length)"

    rdms = [
        self.one_body_reduced_density_matrix(idx=idx, do_tracing=True, inplace_mutation=True)
        for idx in indices
    ]
    rdms = torch.stack(rdms)  # (length, 2, 2)
    eigenvalues = torch.linalg.eigvalsh(rdms)  # (length, 2)
    probs = eigenvalues / eigenvalues.sum(dim=1, keepdim=True)  # (length, 2)
    probs[probs < eps] = eps
    entropies = -(probs * torch.log(probs)).sum(dim=1)  # (length,)
    return entropies

# %% ../../5-2.ipynb 13
from einops import rearrange


@patch
def two_body_reduced_density_matrix_(
    self: MPS, qubit_idx0: int, qubit_idx1: int, return_matrix: bool = False
) -> torch.Tensor:
    assert 0 <= qubit_idx0 < qubit_idx1
    self.center_orthogonalization_(qubit_idx0, mode="qr", normalize=True)

    tensor_left = self._mps[qubit_idx0]
    product = einsum(
        tensor_left.conj(),
        tensor_left,
        "left physical_conj right_conj, left physical right -> physical_conj physical right_conj right",
    )

    for idx in range(qubit_idx0 + 1, qubit_idx1):
        tensor_i = self._mps[idx]
        product = einsum(
            product,
            tensor_i.conj(),
            tensor_i,
            "i0_physical_conj i0_physical left_conj left, left_conj physical right_conj, left physical right -> i0_physical_conj i0_physical right_conj right",
        )

    tensor_right = self._mps[qubit_idx1]
    rdm = einsum(
        product,
        tensor_right.conj(),
        tensor_right,
        "i0_physical_conj i0_physical left_conj left, left_conj i1_physical_conj right, left i1_physical right -> i0_physical i1_physical i0_physical_conj i1_physical_conj ",
    )

    if return_matrix:
        return rearrange(rdm, "a b c d -> (a b) (c d)")
    else:
        return rdm