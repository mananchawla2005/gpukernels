# Day 54: Triplet Loss CUDA Implementation

This project implements the Triplet Loss function in CUDA, a distance-based loss function commonly used in metric learning and face recognition tasks to learn embeddings where similar samples are closer together and dissimilar samples are farther apart.

## What is Triplet Loss?

Triplet Loss is an optimization technique that:

1. Takes three inputs: anchor, positive (similar to anchor), and negative (dissimilar to anchor) samples
2. Minimizes the distance between anchor and positive samples
3. Maximizes the distance between anchor and negative samples
4. Uses a margin (alpha) to enforce a minimum separation between positive and negative pairs
5. Commonly used in face recognition, image retrieval, and similarity learning tasks

## How It Works

### Core Algorithm

The triplet loss is calculated as:
```
L = max(0, d(anchor, positive) - d(anchor, negative) + alpha)
```
where:
- d(x,y) is the squared Euclidean distance between x and y
- alpha is the margin that enforces separation
- max(0,_) ensures the loss is always non-negative

### Key Components

1. **Distance Calculation**:
   - Computes squared Euclidean distance between anchor-positive and anchor-negative pairs
   - Uses parallel reduction for efficiency

2. **Loss Computation**:
   - Applies margin-based comparison using alpha parameter
   - Uses atomic operations to accumulate loss across threads
   - Enforces minimum separation between positive and negative examples

3. **Memory Management**:
   - Efficiently transfers data between host and device
   - Uses contiguous memory layout for coalesced access
   - Properly allocates and frees GPU memory

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler compatible with CUDA
- Basic understanding of metric learning concepts

## Setup

```bash
# Compile the CUDA code
nvcc -o triplet_loss triplet_loss.cu

# Run the executable
./triplet_loss
```

## Implementation Details

The implementation consists of:

1. **CUDA Kernel**: 
   - Parallelizes loss computation across samples
   - Uses atomic operations for thread-safe accumulation
   - Optimized for GPU execution

2. **Host Interface**:
   - Handles memory management between CPU and GPU
   - Configures kernel launch parameters
   - Provides clean interface for loss computation

## External References

[Triplet Loss Paper](https://arxiv.org/abs/1503.03832)  
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
[FaceNet: A Unified Embedding for Face Recognition](https://arxiv.org/abs/1503.03832)
