#include <cuda_runtime.h>

__global__ void custom_kernel(float *input, float *output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * 2.0f; // Example: multiply by 2
  }
}

extern "C" void custom_kernel_launcher(float *input, float *output, int size) {
  int block_size = 256;
  int grid_size = (size + block_size - 1) / block_size;
  custom_kernel<<<grid_size, block_size>>>(input, output, size);
}
