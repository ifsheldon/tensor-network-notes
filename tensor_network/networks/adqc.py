"""初始化："""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../3-2.ipynb.

# %% auto 0
__all__ = ['ADQCNet', 'probabilities_adqc_classifier', 'calc_accuracy']

# %% ../../3-2.ipynb 2
import torch

# %% ../../3-5.ipynb 12
from torch import nn
from typing import Literal, Tuple, List
from ..tensor_gates.modules import ADQCGate


class ADQCNet(nn.Module):
    """
    A simple ADQC network.
    """

    def __init__(
        self,
        *,
        num_qubits: int,
        num_layers: int,
        gate_pattern: Literal["brick", "stair"],
        identity_init: bool = False,
        double_precision: bool = False,
    ):
        """
        Args:
            num_qubits (int): The number of qubits in the network.
            num_layers (int): The number of layers in the network.
            gate_pattern (Literal["brick", "stair"]): The pattern of the gates in the network.
            identity_init (bool): Whether to initialize the gates with identity matrix + random noise.
            double_precision (bool): Whether to use double precision for the gates.
        """
        super().__init__()
        target_positions = self.calc_gate_target_qubit_positions(gate_pattern, num_qubits)
        gates = []

        for layer_idx in range(num_layers):
            for target_qubit_indices in target_positions:
                gate = ADQCGate(
                    batched_input=True,
                    target_qubit=list(target_qubit_indices),
                    gate_name=f"ADQC-{layer_idx}-{target_qubit_indices}",
                    identity_init=identity_init,
                    double_precision=double_precision,
                )
                gates.append(gate)

        self.net = nn.Sequential(*gates)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) % 2 == 1, (
            "x must have an odd number of dimensions, as the first dimension is the batch dimension"
        )
        return self.net(x)

    @staticmethod
    def calc_gate_target_qubit_positions(
        gate_pattern: Literal["brick", "stair"], num_qubits: int
    ) -> List[Tuple[int, int]]:
        assert gate_pattern in ["brick", "stair"], (
            f"""gate_pattern must be either "brick" or "stair", but got {gate_pattern}"""
        )
        assert num_qubits > 0, "number of qubits must be greater than 0"
        target_positions = []
        if gate_pattern == "stair":
            for p in range(num_qubits - 1):
                target_positions.append((p, p + 1))
        else:  # brick
            p = 0
            while p < num_qubits - 1:
                target_positions.append((p, p + 1))
                p += 2
            p = 1
            while p < num_qubits - 1:
                target_positions.append((p, p + 1))
                p += 2
        return target_positions

# %% ../../3-5.ipynb 26
from numpy import ceil, log2


def probabilities_adqc_classifier(qubit_states: torch.Tensor, num_classes: int):
    """
    Compute normalized class probabilities from qubit states for a quantum classifier.

    Args:
        qubit_states (torch.Tensor): Tensor containing the quantum states. The tensor should have
                                     an odd number of dimensions with the first dimension being
                                     the batch size.
        num_classes (int): Number of classes for classification. Must be >= 2.

    Returns:
        torch.Tensor: Normalized probabilities for each class. Shape: (batch_size, num_classes)

    Notes:
        - The function takes the last log2(num_classes) qubits and uses their measured
          probabilities as the classifier's output.
        - Only the first `num_classes` base states are used for classification.
        - Probabilities are normalized to sum to 1 for each sample.
    """
    DELTA = 1e-10
    assert qubit_states.ndimension() % 2 == 1
    assert num_classes >= 2, "number of classes must be greater than 2"
    num_qubit_required = int(ceil(log2(num_classes)))
    assert qubit_states.ndimension() >= num_qubit_required + 1
    batch_size = qubit_states.shape[0]
    # take the last `num_qubit_required` qubits and use their measured probabilities as the classifier's output
    qubit_states = qubit_states.reshape(batch_size, -1, 2**num_qubit_required)
    # use the first `num_classes` base states of the `num_qubit_required` qubits as the classifier's output
    substates = qubit_states[:, :, :num_classes]  # (batch_size, 2**other_qubit_num, num_classes)
    probabilities = (substates * substates.conj()).real
    probabilities_of_classes = torch.sum(probabilities, dim=1)  # (batch_size, num_classes)
    prob_norm = torch.sum(probabilities_of_classes, dim=1, keepdim=True) + DELTA  # (batch_size, 1)
    normalized_class_probabilities = probabilities_of_classes / prob_norm
    return normalized_class_probabilities

# %% ../../3-5.ipynb 28
def calc_accuracy(probabilities: torch.Tensor, targets: torch.Tensor) -> float:
    assert probabilities.ndimension() == 2
    assert targets.ndimension() == 1
    assert probabilities.shape[0] == targets.shape[0]
    predicted_classes = torch.argmax(probabilities, dim=1)
    return (predicted_classes == targets).float().mean().item()