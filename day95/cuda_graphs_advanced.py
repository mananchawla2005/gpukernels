import torch
import torchvision.models as models
import time

def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    
    # Model setup with larger batch
    model = models.resnet50(pretrained=True).to(device).eval()
    batch_size = 256  # Larger batch for better GPU utilization
    static_input = torch.rand((batch_size, 3, 224, 224), 
                            device=device, 
      )  # Pinned memory
    
    # Enable graph memory pooling
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    # Warmup with separate stream
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(5):
            with torch.no_grad():
                model(static_input)
    torch.cuda.synchronize()

    # Baseline timing
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            model(static_input)
    torch.cuda.synchronize()
    baseline_time = (time.perf_counter() - start) * 1000 / 100

    # Graph capture with memory optimization
    torch.cuda.empty_cache()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
            with torch.no_grad():
                static_output = model(static_input)
    
    # Graph timing
    start = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.perf_counter() - start) * 1000 / 100
    
    print(f"Baseline: {baseline_time:.3f}ms")
    print(f"Graph: {graph_time:.3f}ms")
    print(f"Speedup: {baseline_time/graph_time:.2f}x")

if __name__ == "__main__":
    main()