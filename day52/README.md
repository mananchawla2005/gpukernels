# Day 52: ResNet18 Optimization with Fused Add-ReLU CUDA Kernel

This project implements and evaluates performance optimizations for ResNet18 using a custom CUDA kernel that fuses the addition and ReLU operations commonly found in residual blocks.

## Optimization Overview

ResNet architectures rely heavily on residual connections that perform an element-wise addition followed by a ReLU activation. By fusing these operations into a single CUDA kernel, we can:

1. Reduce memory bandwidth requirements by eliminating intermediate storage
2. Decrease kernel launch overhead by reducing the number of CUDA kernel calls
3. Improve cache utilization through operation fusion
4. Minimize data movement between GPU memory and registers

## Performance Improvements

The optimized implementation shows significant performance gains, especially with larger batch sizes:

| Batch Size | Standard Model | With Fused Add-ReLU | Fused Add-ReLU + Fused Batch Conv ReLU | Fused Add-ReLU + Fused Batch Conv ReLU + Fused Conv Batch |
|------------|----------------|---------------------|---------------------------------------|-------------|
| 64         | CPU: 96.903ms<br>CUDA: 111.147ms | CPU: 22.083ms<br>CUDA: 92.403ms | CPU: 12.832ms<br>CUDA: 63.990ms | CPU: 8.942ms<br>CUDA: 39.393ms |

* Best out of n times are reported
Key observations:
- CPU time reduction of 77.2% due to fewer kernel launches and synchronization points
- CUDA execution time reduction of 16.9% from more efficient memory access patterns
- Benefits increase with larger batch sizes (32, 64, etc.) where parallelism can be better utilized

## Implementation Details

The optimization uses a custom CUDA kernel that:

1. Takes two input tensors and produces one output tensor
2. Performs element-wise addition
3. Applies ReLU activation in a single pass
4. Uses efficient CUDA thread mapping to tensor elements

### Key Components

- **CUDA Kernel**: Implements the fused add-ReLU operation with coalesced memory access
- **C++ Binding**: PyBind11 integration for seamless use from Python
- **PyTorch Extension**: Custom extension built using PyTorch's extension mechanism
- **Model Transformation**: Utility to replace standard operations with our optimized version

## Usage

1. Build the extension:
```bash
pip install -e .
```

2. Apply the optimization to a ResNet model:
```python
from ops.fused_ops import optimize_resnet18
import torchvision.models as models

# Load standard model
model = models.resnet18().cuda()

# Apply optimization
optimized_model = optimize_resnet18(model)
```

3. Run the profiling script to compare performance:
```bash
python resnet_profile.py
```

## Technical Details

### Optimization Strategy

The optimization works by:
1. Identifying locations of addition followed by ReLU in the model
2. Replacing these operations with our fused implementation
3. Maintaining the same mathematical behavior while improving performance

## Requirements

- CUDA-capable NVIDIA GPU
- CUDA Toolkit 10.2+
- PyTorch 1.8+
- Python 3.6+

## Future Improvements

Potential future enhancements:
- Extend fusion to include other operations (batch normalization, convolution)
- Optimize for different precision types (FP16, INT8)
- Add support for dynamic shapes and tensor layouts
- Explore multi-GPU parallelism for even larger batch sizes