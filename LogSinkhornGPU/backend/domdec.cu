#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

////////////////////////////
// balancing kernel
////////////////////////////

template <typename scalar_t>
__global__ void balance(
    int B, int C, int N, scalar_t *nu_basic, scalar_t *mass_delta, scalar_t thresh_step)
{
    // Balance domdec cell marginals in the same composite cell
    // Mauro Bonafini, Bernhard Schmitzer, "Domain decomposition for entropy 
    // regularized optimal transport", Section 6.2.
    
    // B is number of OT problems, C number of basic cells per composite cell
    // N is size of box
    // nu_basic are basic cell marginals, size (B, C, N)
    // mass_delta is cell mass imbalance, size (B, C)
    // thresh_step is the minimum delta that we will consider for sending mass
    int b = blockIdx.x * blockDim.x + threadIdx.x; // problem to balance
    if (b >= B)                                    // take care of bigger-than-size indices
        return;
    scalar_t d1, d2, delta, eta, thresh, m;
    int i, j, n, k, index_i, index_j, temp, index_ik, index_jk;
    // Loop over pairs of cells
    for (i = 0; i < C; i++)
    {
        // for (j = i + 1; j < C; j++) // TODO: try upper triangular
        for (j = i+1; j < C; j++)
        {
            // Compute linear index corresponding to (b, i), (b,j)
            index_i = b * C + i;
            index_j = b * C + j;
            // Get pairwise delta, mass to be transported from i to j
            d1 = mass_delta[index_i];
            d2 = mass_delta[index_j];
            delta = min(max(d1, 0.0), max(-d2, 0.0)) - min(max(-d1, 0.0), max(d2, 0.0));
            if (delta < 0)
            {
                delta = -delta; 
                temp = index_j;
                index_j = index_i; 
                index_i = temp;
            }
            if (delta > thresh_step) // if negative see you at the transpose index
            {
                eta = delta;
                // first try to only transfer mass, where nu2 is already > 0
                thresh = 1E-12;
                for (n = 0; n < 2; n++)
                {
                    for (k = 0; k < N; k++)
                    {
                        index_ik = index_i * N + k;
                        index_jk = index_j * N + k;
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

void BalanceKernel( 
    torch::Tensor nu_basic, 
    torch::Tensor mass_delta, 
    torch::Tensor thresh_step) {

// number of elements to process is size of beta, this is, B*M

    // Get nu_basic size
    int B = nu_basic.size(0);
    int C = nu_basic.size(1);
    int N = nu_basic.size(2);

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(nu_basic.scalar_type(), "balance_cuda_kernel", ([&] {
        balance<scalar_t><<<blocks, threads>>>(
            B, C, N,
            nu_basic.data_ptr<scalar_t>(), mass_delta.data_ptr<scalar_t>(),
            thresh_step.item<scalar_t>());
    }));
    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////
// Generalization of basic-to-composite: add with offset
////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void add_with_offsets_2D(
    int B, int C,
    torch::PackedTensorAccessor32<scalar_t, 3> nu_composite,
    torch::PackedTensorAccessor32<scalar_t, 3> nu_basic,
    torch::PackedTensorAccessor32<scalar_t, 2> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> left_in_composite,
    torch::PackedTensorAccessor32<int, 2> left_in_basic,
    torch::PackedTensorAccessor32<int, 2> width_basic,
    torch::PackedTensorAccessor32<int, 2> bottom_in_composite,
    torch::PackedTensorAccessor32<int, 2> bottom_in_basic,
    torch::PackedTensorAccessor32<int, 2> height_basic
)
{
    // Generalization of sparse vector adition to the context of bounding boxes
    // sum_indices[j] indicates the indices of nu_basic that must be 
    // aggregated to render nu_composite[j]. Each of these basic cells may need
    // to be multiplied by a weight `weights[j][k]`. Relative position of basic
    // cells in the composite bounding box extent are represented by *in_basic, 
    // *in_composite, width and height.
    int j = blockIdx.x * blockDim.x + threadIdx.x; // index of comp cell
    if (j >= B)                                    // take care of indices bigger than size
        return;
    scalar_t u;
    int i, lc, lb, bc, bb, w, h;
    for (int k = 0; k < C; k++) // index of basic cell
    {
        i = sum_indices[j][k];
        if (i >= 0) // negative index means do nothing
        {
            // ***_in_composite and ***_basic are relative positions that we 
            // want to copy.
            u = weights[j][k];
            lc = left_in_composite[j][k];
            lb = left_in_basic[j][k];
            w = width_basic[j][k];
            bc = bottom_in_composite[j][k];
            bb = bottom_in_basic[j][k];
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

torch::Tensor AddWithOffsetsKernel_2D(
        torch::Tensor nu_basic, int w, int h,
        torch::Tensor weights, torch::Tensor sum_indices,
        torch::Tensor left_in_composite, torch::Tensor left_in_basic,
        torch::Tensor width_basic,
        torch::Tensor bottom_in_composite, torch::Tensor bottom_in_basic,
        torch::Tensor height_basic) {


    // Init tensor of precomputed size
    int B = sum_indices.size(0);
    int C = sum_indices.size(1);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    torch::Tensor nu_composite = torch::zeros({B, w, h}, nu_basic.options());

    AT_DISPATCH_FLOATING_TYPES(nu_basic.scalar_type(), "add_with_offsets", ([&] {
        add_with_offsets_2D<scalar_t><<<blocks, threads>>>(
            B, C,
            nu_composite.packed_accessor32<scalar_t, 3>(),
            nu_basic.packed_accessor32<scalar_t, 3>(),
            weights.packed_accessor32<scalar_t, 2>(),
            sum_indices.packed_accessor32<int, 2>(),
            left_in_composite.packed_accessor32<int, 2>(),
            left_in_basic.packed_accessor32<int, 2>(),
            width_basic.packed_accessor32<int, 2>(),
            bottom_in_composite.packed_accessor32<int, 2>(),
            bottom_in_basic.packed_accessor32<int, 2>(),
            height_basic.packed_accessor32<int, 2>());
    }));
    cudaDeviceSynchronize();

    return nu_composite;
}



////////////////////////////////////////////////////////
// Generalization of basic-to-composite: add with offset
////////////////////////////////////////////////////////

template <typename scalar_t>
__global__ void add_with_offsets_2D_output_side(
    int B, int C, int wc, int hc, int wb, int hb,
    torch::PackedTensorAccessor32<scalar_t, 3> nu_composite,
    torch::PackedTensorAccessor32<scalar_t, 3> nu_basic,
    torch::PackedTensorAccessor32<scalar_t, 2> weights,
    torch::PackedTensorAccessor32<int, 2> sum_indices,
    torch::PackedTensorAccessor32<int, 2> offsets_b,
    torch::PackedTensorAccessor32<int, 2> offsets_c
)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of comp cell
    int j = index / (w*h);
    if (j >= B)                 // take care of indices bigger than size
        return;
    int x = (index % (w*h)) / h;
    int y = (index % (w*h)) % h;
    scalar_t u;
    int i, xb, yb; // complete
    for (int k = 0; k < C; k++) // index of basic cell in sum indices
    {
        i = sum_indices[j][k];
        if (i >= 0) // negative index means do nothing
        {
            // ***_in_composite and ***_basic are relative positions that we 
            // want to copy.
            u = weights[j][k];
            xb = x + offsets_c[j][0] - offsets_b[i][0];
            yb = y + offsets_c[j][1] - offsets_b[i][1];
            if ((xb >= 0) && (yb >= 0) && (xb < wb) && (yb < hb))
                nu_composite[j][x][y] += u * nu_basic[i][xb][yb];
        }
    }
}

torch::Tensor AddWithOffsetsKernel_2D_OutputSide(
        torch::Tensor nu_basic, int w, int h,
        torch::Tensor weights, torch::Tensor sum_indices,
        torch::Tensor offsets_c,  torch::Tensor offsets_b)
{


    // Init tensor of precomputed size
    int B = sum_indices.size(0);
    int C = sum_indices.size(1);
    // Basic width and height
    int wb = nu_basic.size(1);
    int hb = nu_basic.size(2);
    
    int threads = 256;
    int blocks = (B*w*h + threads - 1) / threads;

    torch::Tensor nu_composite = torch::zeros({B, w, h}, nu_basic.options());

    AT_DISPATCH_FLOATING_TYPES(nu_basic.scalar_type(), "add_with_offsets_output", ([&] {
        add_with_offsets_2D_output_side<scalar_t><<<blocks, threads>>>(
            B, C, w, h, wb, hb,
            nu_composite.packed_accessor32<scalar_t, 3>(),
            nu_basic.packed_accessor32<scalar_t, 3>(),
            weights.packed_accessor32<scalar_t, 2>(),
            sum_indices.packed_accessor32<int, 2>(),
            offsets_c.packed_accessor32<int, 2>(),
            offsets_b.packed_accessor32<int, 2>()
    }));
    cudaDeviceSynchronize();

    return nu_composite;
}
