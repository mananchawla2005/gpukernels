# Day 4:
# TASK1: RGB IMAGE TO BLURRED IMAGE USING AVG POOL
A CUDA kernel to convert an RGB image to blurred image by taking average of surrounding pixels.

# TASK2: RGB IMAGE TO GAUSSIAN BLURRED IMAGE
A CUDA kernel implementing Gaussian blur filter, which applies weighted averaging based on distance from center pixel.

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

## Testing and Usage Example
Run the test scripts to verify the installation:

### Simple Blur
```python
python simple_blur/test.py
```

### Gaussian Blur
```python
python gaussian_blur/test.py
```

The test scripts will:
1. Load an RGB image (`../day03/dog.jpeg`)
2. Convert it to blurred using the respective CUDA kernels via PyTorch bindings
3. Save the resulting images as:
   - `./simple_blur/dog_blurred.jpeg`
   - `./gaussian_blur/dog_gaussian_blurred.jpeg`

### Example Results
Original Image:
![alt text](../day03/dog.jpeg)

Simple Blur:
![alt text](simple_blur/dog_blurred.jpeg)

Gaussian Blur:
![alt text](gaussian_blur/dog_gaussian_blurred.jpeg)

**Note**: Ensure you're in the activated virtual environment when running the code.