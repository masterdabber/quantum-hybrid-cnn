import torch
import torch.nn as nn
import numpy as np

from quantum_layers import quantum_net, n_qubits, q_depth

class DressedQuantumNet(nn.Module):
    def __init__(self, input_dim=512, num_classes=10):
        super(DressedQuantumNet, self).__init__()

        self.pre_net = nn.Linear(input_dim, n_qubits)
        self.q_params = nn.Parameter(
            0.01 * torch.randn(q_depth * n_qubits)
        )
        self.post_net = nn.Linear(n_qubits, num_classes)

    def forward(self, input_features):
        pre_out = self.pre_net(input_features)
        q_in = torch.tanh(pre_out) * np.pi / 2.0

        q_out = []
        for elem in q_in:
            q_result = torch.tensor(quantum_net(elem, self.q_params), device=elem.device, dtype=torch.float32)
            q_out.append(q_result)

        q_out = torch.stack(q_out)
        return self.post_net(q_out)