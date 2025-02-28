import torch
import triton
import triton.language as tl

def batch_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, h: int, w: int, epsilon: float):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.shape[0]
    grid = lambda meta: (triton.cdiv(n_elements*h*w, 256),)
    batch_norm_kernel[grid](x, output, gamma, beta, n_elements, h, w, epsilon, bs=1024)
    return output

@triton.jit
def batch_norm_kernel(x_ptr, output_ptr, gamma, beta, n_elements, h, w, epsilon, bs: tl.constexpr):
    blockId = tl.program_id(0)
    offsets = blockId*bs+tl.arange(0, bs)
    mask = offsets<n_elements*h*w
    width_offset = offsets%w
    height_offset = (offsets // w) % h
    mean = tl.zeros([bs], dtype=tl.float32)
    for i in range(0, n_elements):
        mean+= tl.load(x_ptr+i*h*w+height_offset*w+width_offset, mask=mask, other=0.0)
    mean /= n_elements

    variance = tl.zeros([bs], dtype=tl.float32)
    for i in range(0, n_elements):
        diff = tl.load(x_ptr+i*h*w+height_offset*w+width_offset, mask=mask, other=0.0) - mean
        variance += diff*diff
    variance /= n_elements

    x = tl.load(x_ptr+offsets, mask=mask)
    output = tl.load(gamma+height_offset*w+width_offset) * (x-mean) / tl.sqrt(variance+epsilon) + tl.load(beta+height_offset*w+width_offset)
    tl.store(output_ptr+offsets, output, mask=mask)


def test_batch_norm():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    
    # Test parameters
    batch_size = 4
    height = 32
    width = 32
    epsilon = 1e-5

    # Create test data
    x = torch.randn(batch_size, height, width).cuda()
    gamma = torch.ones(height, width).cuda()  # Initialize to 1
    beta = torch.zeros(height, width).cuda()   # Initialize to 0

    # Run your implementation
    triton_output = batch_norm(x, gamma, beta, height, width, epsilon)

    # Reshape x so that each pixel position becomes a channel.
    x_ref = x.view(batch_size, height * width, 1, 1)
    gamma_ref = gamma.view(height * width)
    beta_ref = beta.view(height * width)
    
    torch_output = torch.nn.functional.batch_norm(
        x_ref,
        running_mean=None,
        running_var=None,
        weight=gamma_ref,
        bias=beta_ref,
        training=True,
        momentum=0,
        eps=epsilon
    ).view(batch_size, height, width)

    # Compare results
    print(triton_output)
    print(torch_output)
    torch.testing.assert_close(
        triton_output,
        torch_output,
        rtol=1e-3,
        atol=1e-3,
        msg="Triton batch norm doesn't match PyTorch's implementation"
    )


def test_different_sizes():
    # Test with different input sizes
    sizes = [
        (2, 16, 16),
        (8, 32, 32),
        (16, 64, 64)
    ]
    
    for batch_size, height, width in sizes:
        x = torch.randn(batch_size, height, width).cuda()
        gamma = torch.ones(height, width).cuda()
        beta = torch.zeros(height, width).cuda()
        epsilon = 1e-5

        # Should not raise any errors
        output = batch_norm(x, gamma, beta, height, width, epsilon)
        assert output.shape == x.shape

if __name__ == "__main__":
    # Run tests
    test_batch_norm()
    test_different_sizes()
    print("All tests passed!")

