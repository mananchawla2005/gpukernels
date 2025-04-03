from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('resnet_kernels', [
        './fused_add_relu_binding.cpp',
        './fused_add_relu.cu',
    ])
]

setup(
    name='fused_add_relu',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)