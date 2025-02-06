# Day 3:
# TASK: CUDA RGB to Grayscale Conversion with Python Bindings

A CUDA kernel to convert an RGB image to grayscale, wrapped with PyTorch C++ extensions for use in Python.

## Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.6+ with pip
- PyTorch installed
- A C++ compiler compatible with your Python version

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

## Testing
Run the test script to verify the installation:

```python
python test.py
```

The test script will verify the result for a dummy array of data.

## Usage Example
Run the test_with_image script to convert the image of the dog to grayscale:

```python
python test_with_image.py
```

The test script will:
1. Load an RGB image (`./dog.jpeg`)
2. Convert it to grayscale using the CUDA kernel via PyTorch bindings
3. Save the resulting image as `./dog_grayscale.jpeg`

![alt text](dog.jpeg)
![alt text](dog_grayscale.jpeg)

**Note**: Ensure you're in the activated virtual environment when running the code.
