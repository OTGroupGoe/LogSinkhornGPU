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
    int index = blockIdx.x * blockDim.x + threadIdx.x; // linear index for beta, corresponds to cartesian (b, i)
    int b = index / M;
    int i = index % M;
    if (b >= B)
    { // take care of bigger-than-size indices
        return;
    }
    dx = dx * dx;     // turn dx to dx^2 (saves multiplications below)
    Dtype m = -1e30f; // Initialize max for logsumexp stabilization
    // Compute max in first pass. Proved to be faster than online + update
    for (int j = 0; j < N; j++)
    {
        m = max(m, alpha[b * N + j] - (i - j) * (i - j) * dx);
    }
    // Compute stabilized logsumexp
    Dtype s = 0.0f;
    for (int j = 0; j < N; j++)
    {
        s += exp(alpha[b * N + j] - (i - j) * (i - j) * dx - m);
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

//////////////////////////////////////////////////////////////////////
// inner newton method kernel for unbalanced domdec with KL divergence
//////////////////////////////////////////////////////////////////////

__global__ void inner_newton(
    int n_iter, float tol, int N, float *t, float eps, float lam,
    float *lognu, float *lognu_nJ, float *logKTu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // float lameps = lam/eps;
    float epslam = eps / lam;
    float vmin, vmax;
    float lse, g, g_prime;
    float err = tol + 1.0f;

    int i = 0;
    while (i < n_iter && err > tol)
    {
        i++;
        // logsumexp for 2 values == logaddexp
        // vmin = min(lognu_nJ[idx], -lameps*t[idx]);
        // vmax = max(lognu_nJ[idx], -lameps*t[idx]);
        vmin = min(lognu_nJ[idx], -lam * t[idx] / eps);
        vmax = max(lognu_nJ[idx], -lam * t[idx] / eps);
        lse = vmax + log1p(exp(vmin - vmax));

        // compute g(t[idx]) and g'(t[idx])
        g = lse - t[idx] - lognu[idx] - epslam * logKTu[idx];
        
        // g_prime = -lameps/(1 + exp(lameps*t[idx] + lognu_nJ[idx])) - 1;
        g_prime = -lam / (eps * (1 + exp(lam * t[idx] / eps + lognu_nJ[idx]))) - 1;

        // update t
        t[idx] = t[idx] - g / g_prime;
        err = abs(g);
    }
}

void InnerNewtonCUDAKernel(
    int n_iter, float tol, int N, float *t, float eps, float lam,
    float *lognu, float *lognu_nJ, float *logKTu)
{
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    inner_newton<<<numBlocks, blockSize>>>(
        n_iter, tol, N, t, eps, lam,
        lognu, lognu_nJ, logKTu);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(
            Formatter() << "CUDA kernel failed : " << std::to_string(err));
}

