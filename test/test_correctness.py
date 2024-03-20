import torch
import sys
from LogSinkhornGPU import *
import time

n = 32
B = 1
eps = 4.0

M1 = n - 2
M2 = n - 1
N1 = n
N2 = n + 1

M = M1*M2
N = N1*N2

for dtype in [torch.float32, torch.float64]:
    errors = torch.zeros((4, 2))
    options = dict(device = "cuda", dtype = dtype)

    mu = torch.rand(B, M1, M2, **options) + 1
    nu = torch.rand(B, N1, N2, **options) + 1
    mu = mu/torch.sum(mu, dim = (1,2), keepdim = True)
    nu = nu/torch.sum(nu, dim = (1,2), keepdim = True)

    x1 = torch.arange(0,1,1/M1, **options) + 1.0
    x2 = torch.arange(0,1,1/M2, **options) - 0.5
    y1 = torch.arange(0,1,1/N1, **options) + 1.5
    y2 = torch.arange(0,1,1/N2, **options) - 1.0

    # Get point clouds for point-cloud solvers
    X = torch.cartesian_prod(x1, x2) # Same as meshgrid, raveling and concatting
    Y = torch.cartesian_prod(y1, y2)
    X = X[None, :, :] # Add batch dimension
    Y = Y[None, :, :] # Add batch dimension

    # Dense solver
    # cost must be of shape (B, M, N), or (1, M, N) if it is the same for all problems
    C_dense = torch.sum((X.reshape(1, -1, 1, 2) - Y.reshape(1, 1, -1, 2))**2, dim = 3).reshape(1, M, N)
    # marginals must be of shape (B, M) and (B, N)
    solver_dense = LogSinkhornTorch(mu.reshape(B, -1), nu.reshape(B, -1), C_dense, eps)
    solver_dense.iterate(1)
    a_ref_1 = solver_dense.alpha.clone().reshape(B, M1, M2)
    solver_dense.iterate_until_max_error()
    a_ref_N = solver_dense.alpha.clone().reshape(B, M1, M2)


    # Torch grid
    # TorchImage requires cost along each dimension
    C1 = (x1.reshape(-1,1) - y1.reshape(1, -1))**2
    C2 = (x2.reshape(-1,1) - y2.reshape(1, -1))**2
    C_grid = (C1, C2)
    # mu and nu must have the "geometric" shape, i.e. (B, M1, M2), (B, N1, N2)
    solver_grid = LogSinkhornTorchImage(mu, nu, C_grid, eps)

    # Error 1 iter
    solver_grid.iterate(1)
    a_grid = solver_grid.alpha.reshape(B, M1, M2)
    errors[0,0] = torch.sum(torch.abs(a_ref_1 - a_grid))/(B*M)
    # Error upon convergence
    solver_grid.iterate_until_max_error()
    a_grid = solver_grid.alpha.reshape(B, M1, M2)
    errors[0,1] = torch.sum(torch.abs(a_ref_N - a_grid))/(B*M)

    # For the keops solver the cost is encoded by the points cloud coordinates themselves. 
    # Then the cost entries are computed online
    C_keops = (X.contiguous(), Y.contiguous())
    solver_keops = LogSinkhornKeops(mu.reshape(B, -1), nu.reshape(B, -1), C_keops, eps)
    # Error 1 iter
    solver_keops.iterate(1)
    a_keops = solver_keops.alpha.reshape(B, M1, M2)
    errors[1,0] = torch.sum(torch.abs(a_ref_1 - a_keops))/(B*M)
    # Error upon convergence
    solver_keops.iterate_until_max_error()
    a_keops = solver_keops.alpha.reshape(B, M1, M2)
    errors[1,1] = torch.sum(torch.abs(a_ref_N - a_keops))/(B*M)

    # Keops grid
    # For the keops grid solver we pass the coordinates of the grid accross each dimension
    # The cost is then computed online
    xs = (x1,x2)
    ys = (y1,y2)
    solver_keops_image = LogSinkhornKeopsImage(mu, nu, (xs, ys), eps)
    # Error 1 iter
    solver_keops_image.iterate(1)
    a_keops_image = solver_keops_image.alpha.reshape(B, M1, M2)
    errors[2,0] = torch.sum(torch.abs(a_ref_1 - a_keops_image))/(B*M)
    # Error upon convergence
    solver_keops_image.iterate_until_max_error()
    a_keops_image = solver_keops_image.alpha.reshape(B, M1, M2)
    errors[2,1] = torch.sum(torch.abs(a_ref_N - a_keops_image))/(B*M)

    # CUDA
    # For the cuda solver we just pass the pixelsize, i.e. the distance between consecutive grid points. 
    # For this to produce the correct solution, the grids must start at the origin and have regular spacing
    # between consecutive points. If the grids do not start at the origin, some preprocessing and post processing
    # is needed; this is detailed in the example `cuda_solver_offset`.
    x1 = x1.reshape(1, -1)
    x2 = x2.reshape(1, -1)
    y1 = y1.reshape(1, -1)
    y2 = y2.reshape(1, -1)
    solver_cuda_image = LogSinkhornCudaImageOffset(mu, nu, ((x1, x2), (y1, y2)), eps)
    # Error 1 iter
    solver_cuda_image.iterate(1)
    a_cuda_image = solver_cuda_image.alpha.reshape(B, M1, M2)
    errors[3,0] = torch.sum(torch.abs(a_ref_1 - a_cuda_image))/(B*M)
    # Error upon convergence
    solver_cuda_image.iterate_until_max_error()
    a_cuda_image = solver_cuda_image.alpha.reshape(B, M1, M2)
    errors[3,1] = torch.sum(torch.abs(a_ref_N - a_cuda_image))/(B*M)
    
    assert torch.all(errors < 1e-5), "some errors bigger than tolerance"