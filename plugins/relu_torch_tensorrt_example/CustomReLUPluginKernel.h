#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA kernel declarations
void customReLU(const float* input, float* output, int n, cudaStream_t stream);
void customReLUFP16(const __half* input, __half* output, int n, cudaStream_t stream);