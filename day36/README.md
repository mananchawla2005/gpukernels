# Day 36: Dynamic Tanh with Weights and Bias

A CUDA kernel implementing a dynamic version of the hyperbolic tangent (tanh) activation function with additional parameters for greater flexibility in neural networks: alpha scaling factor, element-wise weights, and bias terms.

## What is Dynamic Tanh?

This enhanced version extends the standard tanh activation function with three customizable parameters:

1. **Alpha (α)** - Controls the steepness of the tanh curve
2. **Weight (w)** - Element-wise multiplication after tanh activation 
3. **Bias (b)** - Element-wise addition after weighted tanh

The function computes:
  output = tanh(α × input) × weight + bias

## Use Cases

- **Fine-tuning neural network dynamics** with parameterized activation functions
- **Feature scaling** by adjusting the alpha parameter
- **Creating custom activation behaviors** with weight and bias terms
- **Transfer learning** where activation adaptation is needed

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- C++ compiler compatible with your Python version

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
Run the test script to verify the implementation:

```bash
python test.py
```

The test validates three key scenarios:
- Basic functionality with default parameters
- Custom alpha, weights and bias values
- Large random tensors for performance testing

**Note**: Ensure you're in the activated virtual environment when running the code.

## External References

[Tanh Function Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent)  
[PyTorch Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html)