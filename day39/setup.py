from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('gpu_kernels', [
        './tiled_matmul_coarsened_binding.cpp',
        './tiled_matmul_coarsened.cu',
    ])
]

setup(
    name='tiled_matmul_coarsened',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)