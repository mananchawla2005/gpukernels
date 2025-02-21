from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('gpu_kernels', [
        './nf4_binding.cpp',
        './nf4.cu',
    ])
]

setup(
    name='nf4',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)