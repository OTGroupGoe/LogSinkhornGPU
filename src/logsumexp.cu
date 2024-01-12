#include "src/utils.hpp"
#include <math.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

////////////////////////////
// logsumexp kernel
////////////////////////////

// Compiling this with --use_fast_math we get some noticeable performance improvement
template <typename Dtype>
__global__ void logsumexp(int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx)
{
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)
    int index = blockIdx.x * blockDim.x + threadIdx.x; // linear index for beta, corresponds to cartesian (b, j)
    int b = index / M;
    int j = index % M;
    if (b >= B)
    { // take care of bigger-than-size indices
        return;
    }
    dx = dx * dx;     // turn dx to dx^2 (saves multiplications below)
    Dtype m = -1e30f; // Initialize max for logsumexp stabilization
    // Compute max in first pass. Proved to be faster than online + update
    for (int i = 0; i < N; i++)
    {
        m = max(m, alpha[b * N + i] - (j - i) * (j - i) * dx);
    }
    // Compute stabilized logsumexp
    Dtype s = 0.0f;
    for (int i = 0; i < N; i++)
    {
        s += exp(alpha[b * N + i] - (j - i) * (j - i) * dx - m);
    }
    // Remove stabilization
    beta[index] = log(s) + m;
}

template <typename Dtype>
void LogSumExpCUDAKernel(
    int B, int M, int N, Dtype *alpha, Dtype *beta, Dtype dx
)
{
    // number of elements to process is size of beta, this is, B*M
    int blockSize = 256;
    int numBlocks = (B * M + blockSize - 1) / blockSize;
    logsumexp<Dtype><<<numBlocks, blockSize>>>(B, M, N, alpha, beta, dx);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(Formatter()
                                 << "CUDA kernel failed : " << std::to_string(err));
}

// Instantiate
template void LogSumExpCUDAKernel<float>(
    int B, int M, int N, float *alpha, float *beta, float dx
);
template void LogSumExpCUDAKernel<double>(
    int B, int M, int N, double *alpha, double *beta, double dx
);

////////////////////////////
// logsumexp arbitrary dy
////////////////////////////

// Compiling this with --use_fast_math we get some noticeable performance improvement
template <typename Dtype>
__global__ void logsumexp_xyeps(
    int B, int M, int N, Dtype *alpha, Dtype *beta, 
    Dtype dx, Dtype dy, Dtype eps
) {
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)
    int index = blockIdx.x * blockDim.x + threadIdx.x; // linear index for beta, corresponds to cartesian (b, j)
    int b = index / M;
    int j = index % M;
    if (b >= B)
    { // take care of bigger-than-size indices
        return;
    }
    Dtype m = -1e30f; // Initialize max for logsumexp stabilization
    // Compute max in first pass. Proved to be faster than online + update
    for (int i = 0; i < N; i++)
    {
        m = max(m, alpha[b*N+i] - (i*dx - j*dy)*(i*dx - j*dy)/eps);
    }
    // Compute stabilized logsumexp
    Dtype s = 0.0f;
    for (int i = 0; i < N; i++)
    {
        s += exp(alpha[b*N+i] - (i*dx - j*dy)*(i*dx - j*dy)/eps - m);
    }
    // Remove stabilization
    beta[index] = log(s) + m;
}

template <typename Dtype>
void LogSumExpCUDAKernel_xyeps(
    int B, int M, int N, Dtype *alpha, Dtype *beta, 
    Dtype dx, Dtype dy, Dtype eps)
{
    // number of elements to process is size of beta, this is, B*M
    int blockSize = 256;
    int numBlocks = (B * M + blockSize - 1) / blockSize;
    logsumexp<Dtype><<<numBlocks, blockSize>>>(B, M, N, alpha, beta, dx, dy, eps);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(Formatter()
                                 << "CUDA kernel failed : " << std::to_string(err));
}

// Instantiate
template void LogSumExpCUDAKernel_xyeps<float>(
    int B, int M, int N, float *alpha, float *beta, 
    float dx, float dy, float eps
);
template void LogSumExpCUDAKernel_xyeps<double>(
    int B, int M, int N, double *alpha, double *beta, 
    double dx, double dy, double eps
);