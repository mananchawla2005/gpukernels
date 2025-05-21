import argparse
import torch
import torchvision.models as models
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18","resnet50","mobilenet"], default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available"); return

    device = torch.device("cuda")
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    # pick model
    if args.model == "resnet18":
        model = models.resnet18(pretrained=True)
    elif args.model == "mobilenet":
        model = models.mobilenet_v2(pretrained=True)
    else:
        model = models.resnet50(pretrained=True)
    model = model.to(device).eval()

    batch_size = args.batch_size
    num_iters = args.iters
    print(f"Model={args.model}, batch={batch_size}, iters={num_iters}")

    static_input = torch.rand((batch_size,3,224,224), device=device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(static_input)

    # baseline
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        with torch.no_grad():
            model(static_input)
    torch.cuda.synchronize()
    baseline_time = (time.time() - start)*1000/num_iters

    # graph
    torch.cuda.empty_cache()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        with torch.no_grad():
            _ = model(static_input)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        g.replay()
    torch.cuda.synchronize()
    graph_time = (time.time() - start)*1000/num_iters

    print(f"Baseline: {baseline_time:.3f}ms")
    print(f"Graph:    {graph_time:.3f}ms")
    print(f"Speedup:  {baseline_time/graph_time:.2f}x")

if __name__ == "__main__":
    main()