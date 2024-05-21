#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__global__ void inner_newton(
    int N, int n_iter, scalar_t tol, scalar_t eps, scalar_t lam,
    scalar_t *t, scalar_t *lognu, scalar_t *lognu_nJ, scalar_t *logKTu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;

    // scalar_t lameps = lam/eps;
    scalar_t epslam = eps / lam;
    scalar_t vmin, vmax;
    scalar_t lse, g, g_prime;
    scalar_t err = tol + 1.0f;

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
    int n_iter, torch::Tensor tol, torch::Tensor eps, torch::Tensor lam,
    torch::Tensor t, torch::Tensor lognu, 
    torch::Tensor lognu_nJ, torch::Tensor logKTu)
{
    int N = t.numel();
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // Dispatch dynamically as a function of t's type. 
    AT_DISPATCH_FLOATING_TYPES(t.scalar_type(), "inner_newton_kernel", ([&] {
            inner_newton<scalar_t><<<blocks, threads>>>(
                N, n_iter, tol.item<scalar_t>(), 
                eps.item<scalar_t>(), lam.item<scalar_t>(), 
                t.data_ptr<scalar_t>(), lognu.data_ptr<scalar_t>(),
                lognu_nJ.data_ptr<scalar_t>(), logKTu.data_ptr<scalar_t>());
        }));
    cudaDeviceSynchronize();
}