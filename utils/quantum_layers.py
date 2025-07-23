import pennylane as qml
from pennylane import numpy as np
import torch

n_qubits = 4
q_depth = 3

dev = qml.device("default.qubit", wires=n_qubits)

def H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
def RY_layer(weights):
    for idx, angle in enumerate(weights):
        qml.RY(angle, wires=idx)
def entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])

@qml.qnode(dev, interface="torch")
def quantum_net(q_input_features, q_weights_flat):
    q_weights = q_weights_flat.reshape(q_depth, n_qubits)
    H_layer(n_qubits)
    RY_layer(q_input_features)
    for layer_idx in range(q_depth):
        entangling_layer(n_qubits)
        RY_layer(q_weights[layer_idx])

    return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]

weight_shapes = {"q_weights_flat": q_depth * n_qubits}

