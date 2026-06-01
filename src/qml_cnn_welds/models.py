"""Model architectures for QML-CNN-Welds."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pennylane as qml  # type: ignore
except ImportError:
    qml = None  # type: ignore


class SimpleCNN(nn.Module):
    """Small CNN for 8 by 8 grayscale image classification."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class HybridQNN(nn.Module):
    """Compact hybrid quantum-classical model backed by PennyLane."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required when model.type is set to 'hybrid'.")

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.num_classes = num_classes

        self.flatten = nn.Flatten()
        self.pre_fc = nn.Linear(8 * 8, n_qubits)
        self.q_weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit
        self.post_fc = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.tanh(self.pre_fc(x)) * torch.pi

        q_out = []
        for sample in x:
            measured = self.circuit(sample, self.q_weights)
            q_out.append(torch.stack(tuple(measured)).to(sample.device))

        return self.post_fc(torch.stack(q_out))
