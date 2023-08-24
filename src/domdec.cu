#include "src/utils.hpp"
#include <math.h>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

////////////////////////////
// balancing kernel
////////////////////////////

template <typename Dtype>
__global__ void balance(
    int B, int C, int N, Dtype *nu_basic, Dtype *mass_delta, Dtype thresh_step)
{
    // B is number of OT problems, C number of basic cells per composite cell
    // N is size of box
    // nu_basic are basic cell marginals, size (B, C, N)
    // mass_delta is cell mass imbalance, size (B, C)
    // thresh_step is the minimum delta that we will consider for sending mass
    int b = blockIdx.x * blockDim.x + threadIdx.x; // problem to balance
    if (b >= B)                                    // take care of bigger-than-size indices
        return;
    Dtype d1, d2, delta, eta, thresh, m;
    // Loop over pairs of cells
    for (int i = 0; i < C; i++)
    {
        // for (int j = i + 1; j < C; j++) // TODO: try upper triangular
        for (int j = 0; j < C; j++)
        {
            // Compute linear index corresponding to (b, i), (b,j)
            int index_i = b * C + i;
            int index_j = b * C + j;
            // Get pairwise delta, mass to be transported from i to j
            d1 = mass_delta[index_i];
            d2 = mass_delta[index_j];
            delta = min(max(d1, 0.0), max(-d2, 0.0)) - min(max(-d1, 0.0), max(d2, 0.0));
            if (delta > thresh_step) // if negative see you at the transpose index
            {
                eta = delta;
                // first try to only transfer mass, where nu2 is already > 0
                thresh = 1E-12;
                for (int n = 0; n < 2; n++)
                {
                    for (int k = 0; k < N; k++)
                    {
                        int index_ik = index_i * N + k;
                        int index_jk = index_j * N + k;
                        if ((nu_basic[index_ik] > 0.) && (nu_basic[index_jk] >= thresh))
                        {
                            // compute mass to be transferred
                            m = min(eta, nu_basic[index_ik]);
                            nu_basic[index_jk] += m;
                            nu_basic[index_ik] -= m;
                            eta -= m;
                        }
                    }
                    // if fist loop was not  sufficient, set thresh to 0
                    if (eta > thresh_step);
                        thresh = 0;
                }
                // Update the deltas
                delta -= eta; // There may be some rounding errors (?)
                mass_delta[index_i] -= delta;
                mass_delta[index_j] += delta;
            }
        }
    }
}

template <typename Dtype>
void BalanceKernel(
    int B, int C, int N, Dtype *nu_basic, Dtype *mass_delta, Dtype thresh_step
)
{
    // number of elements to process is size of beta, this is, B*M
    int blockSize = 256;
    int numBlocks = (B + blockSize - 1) / blockSize;
    balance<Dtype><<<numBlocks, blockSize>>>(B, C, N, nu_basic, mass_delta, thresh_step);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(Formatter()
                                 << "CUDA kernel failed : " << std::to_string(err));
}

// Instantiate
template void BalanceKernel<float>(
    int B, int C, int N, float *nu_basic, float *mass_delta, float thresh_step
);
template void BalanceKernel<double>(
    int B, int C, int N, double *nu_basic, double *mass_delta, double thresh_step
);