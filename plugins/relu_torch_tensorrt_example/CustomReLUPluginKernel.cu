#include "CustomReLUPluginKernel.h"

// CUDA kernel for ReLU
__global__ void customReLUKernel(const float* input, float* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
    }
}

__global__ void customReLUKernelFP16(const __half* input, __half* output, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = __half2float(input[idx]);
        output[idx] = __float2half(val > 0.0f ? val : 0.0f);
    }
}

void customReLU(const float* input, float* output, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    customReLUKernel<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const float*>(input),
        static_cast<float*>(output),
        n
    );
}

void customReLUFP16(const __half* input, __half* output, int n, cudaStream_t stream)
{
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    customReLUKernelFP16<<<gridSize, blockSize, 0, stream>>>(
        static_cast<const __half*>(input),
        static_cast<__half*>(output),
        n
    );
}