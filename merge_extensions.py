import os
import re
from collections import defaultdict

def find_kernel_pairs():
    kernel_pairs = defaultdict(dict)
    
    # Walk through all directories
    for root, _, files in os.walk('.'):
        if re.search(r"day\d+", root):
            # Find .cu and .cpp files
            for file in files:
                if file.endswith('.cu'):
                    kernel_pairs[root]['kernel'] = os.path.join(root, file)
                elif file.endswith('binding.cpp'):
                    kernel_pairs[root]['binding'] = os.path.join(root, file)
    
    return kernel_pairs

def generate_merged_binding(kernel_pairs):
    # Generate merged binding file
    merged_binding = """
#include <torch/extension.h>

// Forward declarations
"""
    
    # Add forward declarations
    for folder, files in kernel_pairs.items():
        if 'kernel' in files:
            kernel_name = os.path.splitext(os.path.basename(files['kernel']))[0]
            merged_binding += f'extern "C" void {kernel_name}(float* q_h, float* attn_out_h, int rows, int cols, int d);\n'
    
    # Start PYBIND11_MODULE
    merged_binding += """
PYBIND11_MODULE(gpu_kernels, m) {
"""
    
    # Add function definitions
    for folder, files in kernel_pairs.items():
        if 'binding' in files:
            with open(files['binding'], 'r') as f:
                content = f.read()
                # Extract the m.def block
                if 'PYBIND11_MODULE' in content:
                    start = content.find('m.def')
                    end = content.find(');', start) + 2
                    merged_binding += f"    {content[start:end]}\n"
    
    merged_binding += "}\n"
    
    # Write merged binding
    with open("merged_binding.cpp", "w") as f:
        f.write(merged_binding)

def generate_merged_setup(kernel_pairs):
    setup_content = """
# filepath: merged_setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

ext_modules = [
    CUDAExtension(
        'gpu_kernels',
        [
            'merged_binding.cpp',
"""
    
    # Add all kernel files
    for folder, files in kernel_pairs.items():
        if 'kernel' in files:
            setup_content += f"            r'{files['kernel']}',\n"
    
    setup_content += """        ]
    )
]

setup(
    name='merged_kernels',
    version=__version__,
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
"""
    
    with open("merged_setup.py", "w") as f:
        f.write(setup_content)

if __name__ == "__main__":
    kernel_pairs = find_kernel_pairs()
    generate_merged_binding(kernel_pairs)
    generate_merged_setup(kernel_pairs)
    print("Generated merged_binding.cpp and merged_setup.py")