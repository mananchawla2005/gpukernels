from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('gpu_kernels', [
        './vec_add_binding.cpp',
        './vec_add_python.cu',
    ])
]

setup(
    name='vec_add',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)

# To build in development mode just run python setup.py build_ext --inplace