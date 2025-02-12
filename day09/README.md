# Day 9:
# TASK: Sigmoid Activation Function Implementation

A CUDA kernel implementing the sigmoid activation function, a fundamental component in neural networks for introducing non-linearity.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY SIGMOID?

Sigmoid function squashes input values into the range [0,1], making it useful for:
- Output layers in binary classification
- Gates in LSTM/GRU networks
- Generating probability values
- Historical importance in neural network development

The sigmoid function is computed as:
  σ(x) = 1 / (1 + e^(-x))

## Setup
1. Create and activate a virtual environment:

```bash
python -m venv myenv
myenv\Scripts\activate  # On Windows
source myenv/bin/activate  # On Unix/Linux
```

2. Build and install the package:

```bash
pip install .
```

## Testing and Usage Example
Run the test script to verify the installation:

```bash
python test.py
```

**Note**: Ensure you're in the activated virtual environment when running the code.

# External References

[Sigmoid Function Wikipedia](https://en.wikipedia.org/wiki/Sigmoid_function)  
[PyTorch Sigmoid Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html)