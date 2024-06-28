import numpy as np
from numba import cuda
import time

@cuda.jit
def add_kernel(x, y, out):
    i = cuda.grid(1)
    if i < x.size:
        out[i] = x[i] + y[i]

# Initialize arrays
N = 10000000
x = np.ones(N, dtype=np.float32)
y = np.ones(N, dtype=np.float32) * 2
out = np.zeros(N, dtype=np.float32)

# Allocate device arrays
d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_out = cuda.to_device(out)

# Configure the blocks
threads_per_block = 256
blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block

# Run kernel
start = time.time()
add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
cuda.synchronize()
end = time.time()

# Copy the result back to the host
d_out.copy_to_host(out)

# Check for errors (all values should be 3.0)
max_error = np.max(np.abs(out - 3.0))
print(f'Max error: {max_error}')
print(f'Time: {end - start}')
