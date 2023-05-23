#include "src/utils.hpp"
#include <math.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

////////////////////////////
// logsumexp kernel
////////////////////////////

// Compiling this with --use_fast_math we get some noticeable performance improvement
__global__ void logsumexp(int B, int M, int N, float *alpha, float *beta, float dx)
{
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)
    int index = blockIdx.x * blockDim.x + threadIdx.x; // linear index for beta, corresponds to cartesian (b, i)
    int b = index / M;
    int i = index % M;
    if (b >= B){ // take care of bigger-than-size indices
        return;
    }
    dx = dx*dx; // turn dx to dx^2 (saves multiplications below)
    float m = -1e30f; // Initialize max for logsumexp stabilization
    // Compute max in first pass. Proved to be faster than online + update
    for (int j = 0; j<N; j++)
    {
        m = max(m, alpha[b*N+j] - (i-j)*(i-j)*dx); 
    }
    // Compute stabilized logsumexp
    float s = 0.0f;
    for (int j = 0; j<N; j++)
    {
        s += exp(alpha[b*N+j] - (i-j)*(i-j)*dx - m);
    }
    // Remove stabilization
    beta[index] = log(s)+m;
}

void LogSumExpCUDAKernel(int B, int M, int N, float *alpha, float *beta, float dx)
{
  // number of elements to process is size of beta, this is, B*M
  int blockSize = 256;
  int numBlocks = (B*M + blockSize - 1) / blockSize; 
  logsumexp<<<numBlocks, blockSize>>>(B, M, N, alpha, beta, dx);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
    throw std::runtime_error(Formatter()
                             << "CUDA kernel failed : " << std::to_string(err));
}