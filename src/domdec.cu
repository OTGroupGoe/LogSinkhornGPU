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
                    if (eta > thresh_step)
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
    int B, int C, int N, Dtype *nu_basic, Dtype *mass_delta, Dtype thresh_step)
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
    int B, int C, int N, float *nu_basic, float *mass_delta, float thresh_step);
template void BalanceKernel<double>(
    int B, int C, int N, double *nu_basic, double *mass_delta, double thresh_step);

////////////////////////////
// basic_to_composite
////////////////////////////
template <typename Dtype>
__global__ void basic_to_composite_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 4> nu_basic,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x; // index of comp cell
    if (k >= B)                                    // take care of indices bigger than size
        return;
    int lc, lb, bc, bb, w, h;
    for (int b = 0; b < C; b++) // index of basic cell
    {
        lc = left_in_composite[k][b];
        lb = left_in_basic[k][b];
        bc = bottom_in_composite[k][b];
        bb = bottom_in_basic[k][b];
        w = width_basic[k][b];
        h = height_basic[k][b];
        // Fill box
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < h; j++)
            {
                nu_composite[k][lc + i][bc + j] += nu_basic[k][b][lb + i][bb + j];
            }
        }
    }
}

template <typename Dtype>
void BasicToCompositeKernel_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 4> nu_basic,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic)
{
    int blockSize = 256;
    int numBlocks = (B + blockSize - 1) / blockSize;
    basic_to_composite_2D<Dtype><<<numBlocks, blockSize>>>(
        B, C, nu_composite, nu_basic,
        left_in_composite, left_in_basic, width_basic,
        bottom_in_composite, bottom_in_basic, height_basic);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(Formatter()
                                 << "CUDA kernel failed : " << std::to_string(err));
}

// Instantiate
template void BasicToCompositeKernel_2D<float>(
    int B, int C,
    torch::PackedTensorAccessor32<float, 3> nu_composite,
    torch::PackedTensorAccessor32<float, 4> nu_basic,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic);

template void BasicToCompositeKernel_2D<double>(
    int B, int C,
    torch::PackedTensorAccessor32<double, 3> nu_composite,
    torch::PackedTensorAccessor32<double, 4> nu_basic,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic);

////////////////////////////////////////////////////////
// Generalization of basic-to-composite: add with offset
////////////////////////////////////////////////////////

template <typename Dtype>
void add_with_offsets_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 3> nu_basic,
    torch::PackedTensorAccessor32<Dtype, 3> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index of comp cell
    if (j >= B)                                    // take care of indices bigger than size
        return;
    Dtype u;
    int i, lc, lb, bc, bb, w, h;
    for (int k = 0; k < C; k++) // index of basic cell
    {
        i = sum_indices[j][k] if (i >= 0) // negative index means do nothing
        {
            u = weights[j][k];
            lc = left_in_composite[j][k];
            lb = left_in_basic[j][k];
            bc = bottom_in_composite[j][k];
            bb = bottom_in_basic[j][k];
            w = width_basic[j][k];
            h = height_basic[j][k];
            // Fill box
            for (int x = 0; x < w; x++)
            {
                for (int y = 0; y < h; y++)
                {
                    nu_composite[j][lc + x][bc + y] += u * nu_basic[i][lb + x][bb + y];
                }
            }
        }
    }
}

template <typename Dtype>
void AddWithOffsetsKernel_2D(
    int B, int C,
    torch::PackedTensorAccessor32<Dtype, 3> nu_composite,
    torch::PackedTensorAccessor32<Dtype, 3> nu_basic,
    torch::PackedTensorAccessor32<Dtype, 2> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic)
{
    int blockSize = 256;
    int numBlocks = (B + blockSize - 1) / blockSize;
    add_with_offsets_2D<Dtype><<<numBlocks, blockSize>>>(
        B, C, nu_composite, nu_basic, weights, sum_indices,
        left_in_composite, left_in_basic, width_basic,
        bottom_in_composite, bottom_in_basic, height_basic);
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
        throw std::runtime_error(Formatter()
                                 << "CUDA kernel failed : " << std::to_string(err));
}

// Instantiate
template void AddWithOffsetsKernel_2D<float>(
    int B, int C,
    torch::PackedTensorAccessor32<float, 3> nu_composite,
    torch::PackedTensorAccessor32<float, 3> nu_basic,
    torch::PackedTensorAccessor32<float, 2> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic);

template void AddWithOffsetsKernel_2D<double>(
    int B, int C,
    torch::PackedTensorAccessor32<double, 3> nu_composite,
    torch::PackedTensorAccessor32<double, 3> nu_basic,
    torch::PackedTensorAccessor32<double, 2> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic);