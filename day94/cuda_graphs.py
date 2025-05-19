import torch
import torch.nn as nn
import time

device = torch.device('cuda')

class LargerMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        return self.seq(x)

model = LargerMLP().to(device)
model.eval()

torch.cuda.synchronize()

with torch.no_grad():
    for _ in range(10):
        dummy = torch.randn(32, 1024, device=device)
        _ = model(dummy)

print("Running benchmarks...")
num_iters = 1000

eager_inputs = [torch.randn(32, 1024, device=device) for _ in range(num_iters)]
eager_output = torch.empty(32, 10, device=device)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for i in range(num_iters):
        _ = model(eager_inputs[i])
torch.cuda.synchronize()
eager_time = time.time() - start
print(f"Eager time: {eager_time:.6f} seconds")

# Setup for CUDA graph
static_input = torch.randn(32, 1024, device=device)
static_output = torch.empty(32, 10, device=device)

# Create graph
g = torch.cuda.CUDAGraph()

# Pre-run to ensure all initializations are done
with torch.no_grad():
    _ = model(static_input)

# Capture graph
with torch.no_grad(), torch.cuda.graph(g):
    static_output = model(static_input)

# Pre-generate the same random inputs for fair comparison
graph_inputs = [torch.randn(32, 1024, device=device) for _ in range(num_iters)]

# Benchmark CUDA graph execution
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for i in range(num_iters):
        # Copy new input data
        static_input.copy_(graph_inputs[i])
        # Run the graph
        g.replay()
        # Result is now in static_output
torch.cuda.synchronize()
graph_time = time.time() - start
print(f"Graph time: {graph_time:.6f} seconds")
print(f"Speedup: {eager_time / graph_time:.2f}x")