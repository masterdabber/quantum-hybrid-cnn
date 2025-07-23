# Quantum-Enhanced Image Classification with Hybrid CNNs

This project integrates classical deep learning with quantum machine learning to build a hybrid image classification model. Using a pretrained ResNet18 as a feature extractor and a variational quantum circuit (VQC) as the processing core, the model is trained to classify images from the CIFAR-10 dataset. The hybrid approach aims to explore potential quantum advantages in feature representation and non-linear separability.

## Project Overview

The core of this project is a three-stage hybrid architecture:
1. Classical CNN Feature Extractor  
   A pretrained ResNet18 model extracts high-level features from CIFAR-10 images.
2. Quantum Processing Layer  
   Features are fed into a variational quantum circuit built using PennyLane and executed on a simulated quantum device.
3. Classical Classification Head  
   The output of the quantum layer is post-processed by fully connected layers for final classification.

## Objectives

- Leverage the power of quantum circuits to process classical CNN features
- Demonstrate a practical application of hybrid quantum-classical neural networks
- Provide an end-to-end pipeline for training and evaluating hybrid models on image data


## Requirements

- Python 3.11
- PyTorch
- torchvision
- PennyLane
- NumPy
- matplotlib
- scikit-learn
- seaborn

> Recommended: Use a virtual environment and install dependencies via `pip install -r requirements.txt` (optional file).

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/quantum_cnn_project.git
cd quantum_cnn_project

# 2. Create a virtual environment (cross-platform)
python -m venv hybrid_cnn

# 3. Activate the virtual environment
# On Windows:
./hybrid_cnn/Scripts/activate

# On macOS/Linux:
# source hybrid_cnn/bin/activate

# 4. Upgrade pip and install core dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pennylane matplotlib seaborn scikit-learn

# 5. (Optional) Verify GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Training Pipeline

1. Feature Extraction  
   Load a pretrained ResNet18 model with `torchvision.models.resnet18`. Freeze early layers and replace the final FC layer to match CIFAR-10.

2. Quantum Circuit  
   - The quantum layer uses a 4-qubit variational circuit:
     - RY embedding
     - Multiple entangling layers of CNOT gates
     - Expectation values of Pauli-Z measurements
   - Implemented with PennyLane and wrapped in a `qml.qnode`

3. Hybrid Integration  
   - Classical feature vector (size 512) is projected to quantum input dimension
   - Quantum outputs (length 4) are passed into a post-processing dense layer

4. Training Script (`train_quantum.py`)  
   - Applies transformations
   - Uses `DeviceDataLoader` for GPU acceleration
   - Trains the model for 15 epochs with SGD and learning rate scheduler
   - Evaluates using confusion matrix and classification report

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion Matrix (visualized using seaborn)

## Results

- Final accuracy: ~75% on CIFAR-10 test set (can vary by run)
- Quantum layer introduces learnable non-linearity through variational parameters

## Key Quantum Techniques Used

- AngleEmbedding and RY rotations
- Variational quantum circuit layers with CNOT entanglement
- Hybrid gradient flow (autograd support via PennyLane)
- Expectation value measurements for quantum-classical interfacing

## Future Work

- Swap AmplitudeEmbedding with AngleEmbedding for interpretability
- Experiment with quantum attention layers or multi-circuit ensembles
- Deploy using FastAPI or Streamlit for interactive predictions
- Use real quantum hardware (IBMQ) for inference trials


## Author

Arya Palanivel  
Sophomore, Computer Science Major  
AI & Quantum Computing Enthusiast

