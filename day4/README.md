# Day 3:
# TASK1: RGB IMAGE TO BLURRED IMAGE CONVERSION USING AVG POOL

A CUDA kernel to convert an RGB image to blurred image by taking average of surrounding pixels.

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
Run the test script to verify the installation:

```python
python test.py
```

The test script will convert the image of the dog to blurred image:

The test script will:
1. Load an RGB image (`../day3/dog.jpeg`)
2. Convert it to blurred using the CUDA kernel via PyTorch bindings
3. Save the resulting image as `./dog_blurred.jpeg`

![alt text](../day3/dog.jpeg)
![alt text](simple_blur/dog_blurred.jpeg)

**Note**: Ensure you're in the activated virtual environment when running the code.
