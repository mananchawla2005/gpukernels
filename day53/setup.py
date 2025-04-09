from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

__version__ = "0.0.1"
cu_files = glob.glob('kernels/*.cu')

ext_modules = [
    CUDAExtension('kokoro_kernels', [
        './bindings.cpp',
    ] + cu_files)
]

setup(
    name='fused_kokoro_kernels',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)