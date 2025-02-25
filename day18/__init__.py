import torch
import os
print(torch.__file__)
dll_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

os.add_dll_directory(dll_path)
from gpu_kernels import *

import gpu_kernels
__all__ = [name for name in dir(gpu_kernels) if not name.startswith('_')]