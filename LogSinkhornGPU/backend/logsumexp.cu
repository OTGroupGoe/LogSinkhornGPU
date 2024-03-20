#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Compiling this with --use_fast_math we get some noticeable performance improvement
template <typename scalar_t>
__global__ void logsumexp(int B, int M, int N, 
    scalar_t *alpha, scalar_t *beta, scalar_t dx, scalar_t dy)
{
    // Compute the online logsumexp of a[b, i] - |x[b, i] - y[b, j]|^2, along
    // index i, for x equispaced points starting in 0 and separation dx, 
    // y equispaced point starting in 0 and separation dy.
    // alpha is input, with size (B, N)
    // beta is output, with size (B, M)

    // linear index for beta, corresponds to cartesian (b, j)
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    int b = index / M;
    int j = index % M;
    // y coordinate corresponding to beta[index]
    scalar_t y = j*dy;
    if (b >= B)
    { // take care of bigger-than-size indices
        return;
    }
    scalar_t m = -1e30f; // Initialize max for logsumexp stabilization
    // Compute max in first pass. Proved to be faster than online + update
    for (int i = 0; i < N; i++)
    {
        m = max(m, alpha[b * N + i] - (y - i*dx) * (y - i*dx));
    }
    // Compute stabilized logsumexp
    scalar_t s = 0.0f;
    for (int i = 0; i < N; i++)
    {
        s += exp(alpha[b * N + i] - (y - i*dx) * (y - i*dx) - m);
    }
    // Remove stabilization
    beta[index] = log(s) + m;
}

torch::Tensor LogSumExpCUDAKernel(
    torch::Tensor alpha, 
    int M, 
    torch::Tensor dx, 
    torch::Tensor dy) {
  
  int B = alpha.size(0);
  int N = alpha.size(1);

  // Init tensor of size (B, M)
  torch::Tensor beta = torch::empty({B, M}, alpha.options());  

  const int threads = 256;
  const int blocks = (B * M + threads - 1) / threads;
  // Dispatch dynamically as a function of alpha's type. 
  AT_DISPATCH_FLOATING_TYPES(alpha.scalar_type(), "logsumexp_cuda_kernel", ([&] {
        logsumexp<scalar_t><<<blocks, threads>>>(
            B, M, N,
            alpha.data_ptr<scalar_t>(), beta.data_ptr<scalar_t>(),
            dx.item<scalar_t>(), dy.item<scalar_t>());
    }));

  cudaDeviceSynchronize();
  return beta;
}
