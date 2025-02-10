# Day 7:
# TASK: GELU (Gaussian Error Linear Unit) Activation Implementation

A CUDA kernel implementing the GELU activation function, commonly used in transformers and modern neural networks.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

## WHY GELU?

GELU (Gaussian Error Linear Unit) has become a popular activation function, especially in transformer architectures like BERT and GPT. The function is defined as:

GELU(x) = 0.5x * (1 + erf(x/√2))

Advantages over ReLU:
- Smoother gradient flow
- Better performance in deep networks
- Non-linear behavior that approximates attention mechanisms

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
Run the test scripts to verify the installation:

```python
python test.py
```

- Additionally we can benchmark the implementation using the benchmark script

**Note**: Ensure you're in the activated virtual environment when running the code.

# External References

[GELU Paper](https://arxiv.org/abs/1606.08415)
[PyTorch GELU Implementation](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
