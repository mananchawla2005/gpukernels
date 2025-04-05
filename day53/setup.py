from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension('kokoro_kernels', [
        './bindings.cpp',
        'kernels/stft.cu'
    ])
]

setup(
    name='fused_kokoro_kernels',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    # for development mode
    package={'': '.'}
)