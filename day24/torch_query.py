import torch
import triton
import triton.language as tl

def query_device_properties():
    print("====================================")
    print("Torch GPU Device Query")
    print("====================================")
    
    
    device_count = torch.cuda.device_count()
    print(f"Number of available GPU devices: {device_count}")
    
    for device_idx in range(device_count):
        print(f"\nDevice {device_idx}: \"{torch.cuda.get_device_name(device_idx)}\"")
        print("------------------------------------")
        
        
        major, minor = torch.cuda.get_device_capability(device_idx)
        print(f"CUDA Capability Major/Minor version number: {major}.{minor}")
        
        
        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
        print(f"Total Global Memory: {total_memory / (1024**3):.2f} GB")
        
        
        is_available = torch.cuda.is_available()
        print(f"CUDA Available: {'Yes' if is_available else 'No'}")
        
        
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        
        print("\nTriton Capabilities:")
        
        print(f"Triton version: {triton.__version__}")
        
        
        if hasattr(triton, 'backends'):
            backends = triton.backends if isinstance(triton.backends, list) else []
            print(f"Available Triton backends: {backends}")
            print(f"CUDA backend available: {'cuda' in backends}")
        else:
            print("Cannot determine Triton backends from this version")
        
        
        try:
            print("\nSupported CUDA capabilities:")
            for capability in triton.runtime.driver.cuda.get_supported_capabilities():
                print(f"  - SM {capability[0]}.{capability[1]}")
        except (AttributeError, ImportError):
            print("Could not retrieve CUDA capabilities through Triton API")
            
        
        if major >= 7:
            print("Tensor Cores: Available")
        else:
            print("Tensor Cores: Not Available")
            
        print("====================================")

if __name__ == "__main__":
    query_device_properties()