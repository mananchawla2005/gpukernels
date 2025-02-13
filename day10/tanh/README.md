# Day 10:
# TASK 1: Tanh Activation Function Implementation

A CUDA kernel implementing the hyperbolic tangent (tanh) activation function, commonly used in neural networks for introducing non-linearity and normalizing values to [-1,1] range.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY TANH?

Tanh function normalizes input values into the range [-1,1], making it useful for:
- Hidden layers in neural networks
- Handling both positive and negative inputs
- Stronger gradients near zero compared to sigmoid
- Zero-centered outputs that help with learning

The tanh function is computed as:
  tanh(x) = (e^x - e^-x)/(e^x + e^-x)

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

[Tanh Function Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)  
[PyTorch Tanh Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)
