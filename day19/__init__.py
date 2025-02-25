import torch
import os
import sys
print(torch.__file__)
dll_path = os.path.join(os.path.dirname(torch.__file__), 'lib')

os.add_dll_directory(dll_path)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from gpu_kernels import self_attention

__all__ = ['self_attention']