from numba import cuda
import torch

## SIMT/SIMD (Single Instruction Multiple Data/Tensor)

@cuda.jit()
def square(in_arr, out_arr):
    tid = cuda.grid(1)
    for i in range(tid, in_arr.size, cuda.blockDim.x * cuda.gridDim.x):
        out_arr[i] = in_arr[i]**2

in_arr = torch.tensor([5, 4, 5, 6, 6, 7, 8, 9, 7, 4, 3, 2, 1, 2])
out_arr = torch.zeros_like(in_arr)
in_np = in_arr.numpy()
out_np = out_arr.numpy()
square[1, 32](in_np, out_np)
print(torch.from_numpy(out_np))


## CuTile (Tile based programming ) -> will be updated when support arrives
# @cuda.tile.jit

