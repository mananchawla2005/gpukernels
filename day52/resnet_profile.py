import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from ops.fused_ops import optimize_resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models
standard_model = models.resnet18().to(device)
optimized_model = models.resnet18().to(device)
optimized_model = optimize_resnet18(optimized_model)

# Create input data
inputs = torch.randn(64, 3, 224, 224).to(device)
warming_inputs = torch.randn(5, 3, 224, 224).to(device)

# Warmup to initialize CUDA and JIT compilation
print("Warming up models...")
with torch.no_grad():
    for _ in range(10):
        standard_model(warming_inputs)
        optimized_model(warming_inputs)
    
    # Clear GPU cache to ensure fair comparison
    torch.cuda.empty_cache()

# Profile standard model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             record_shapes=True, profile_memory=True) as prof_standard:
    standard_model(inputs)

print("Standard ResNet18 Performance:")
print(prof_standard.key_averages().table(sort_by="cuda_time_total", row_limit=2))
# prof_standard.export_chrome_trace("standard_trace.json")

# Clear cache between runs for fair comparison
torch.cuda.empty_cache()

# Profile optimized model
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
             record_shapes=True, profile_memory=True) as prof_optimized:
    optimized_model(inputs)

print("\nOptimized ResNet18 Performance:")
print(prof_optimized.key_averages().table(sort_by="cuda_time_total", row_limit=2))
prof_optimized.export_chrome_trace("optimized_third_trace.json")