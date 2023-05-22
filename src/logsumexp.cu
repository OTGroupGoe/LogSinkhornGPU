#include "src/utils.hpp"
#include <math.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

////////////////////////////
// Simple routines
////////////////////////////

// Compiling this with --use_fast_math we get some noticeable performance improvement
__global__ void logsumexp(int B, int M, int N, float *alpha, float *beta, float dx)
{
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of the reduction, this is (b,i), this is of the result beta
    int b = index / M;
    int i = index % M;
    if (b >= B){ // care for bigger-than-size indices
        return;
    }
    dx = dx*dx; // for just multiplying to the square cost
    float m = -1e30f; // TODO: check initialization
    for (int j = 0; j<N; j++)
    {
        m = max(m, alpha[b*N+j] - (i-j)*(i-j)*dx); // TODO: still to check the squared part
    }
    float s = 0.0f;
    for (int j = 0; j<N; j++)
    {
        s += exp(alpha[b*N+j] - (i-j)*(i-j)*dx - m); // TODO: still to check the squared part
    }
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