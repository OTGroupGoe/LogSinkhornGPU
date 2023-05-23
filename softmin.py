import torch
from pykeops.torch import LazyTensor
from LogSinkhornGPUBackend import LogSumExpCUDA
import math

def log_dens(a):
    """
    Log of a, thresholded at the - infinities. Taken from `geomloss`.
    """
    a_log = a.log()
    a_log[a <= 0] = -10000.0
    return a_log

def batch_dim(a):
    """
    Batch dimension of tensor.
    """
    return a.shape[0]

def geom_dims(a):
    """
    Dimensions folowing the batch dimension.
    """
    return a.shape[1:]

def softmin_torch(h, dim):
    """
    Compute the logsumexp of `C` along the dimension `dim`.
    """ 
    return torch.logsumexp(h, dim = dim, keepdims = True)

def softmin_torch_image(h, C1, C2, eps):
    """
    Compute the logsumexp of h, with respect to the cost C1 along dimension 1,
    and C2 along dimension 2.
    """ 

    B = batch_dim(h)
    M1, N1 = C1.shape
    M2, N2 = C2.shape
    # h is assumed to be of shape (B, N1, N2)    
    h = h.reshape(B, N1, 1, N2).contiguous()                    # (B, N1, 1, N2)
    h = torch.logsumexp(h - C2/eps, dim = 3, keepdims = True)   # (B, N1, M2, 1)
    h = h.permute((0, 2, 3, 1)).contiguous()                    # (B, M2, 1, N1)
    h = torch.logsumexp(h - C1/eps, dim = 3, keepdims = True)   # (B, M2, M1, 1)
    h = h.permute((0, 2, 1, 3)).reshape(B, M1, M2).contiguous() # (B, M1, M2)
    return h

def softmin_keops(h, x, y, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the variable j.
    Inspired by `geomloss`.
    """ 
    B = batch_dim(h)
    xi = LazyTensor(x[:, :, None, :])
    yj = LazyTensor(y[:, None, :, :])
    Cij = ((xi - yj)**2).sum(-1)
    hj = LazyTensor(h[:, None, :, None])
    return (hj-Cij/eps).logsumexp(2).view(B, -1)

def softmin_keops_line(h, x, y, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the variable j.
    `x` and `y` are unidimensional vectors. Inspired by `geomloss`.
    """ 
    B = batch_dim(h)
    xi = LazyTensor(x.view(1, -1, 1, 1)) # If we remove the last dimension keops freaks out
    yj = LazyTensor(y.view(1, 1, -1, 1))
    Cij = (xi - yj)**2
    hj = LazyTensor(h.view(B, 1, -1, 1))
    return (hj-Cij/eps).logsumexp(2).view(B, -1)

def softmin_keops_image(h, xs, ys, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the variable j, by
    performing "line" logsumexps. Inspired by `geomloss`.
    """ 
    B = batch_dim(h)
    x1, x2 = xs
    y1, y2 = ys
    M1, M2, N1, N2 = len(x1), len(x2), len(y1), len(y2)  
    h = h.reshape(B*N1, 1, N2).contiguous()                                         # (B*N1, 1, N2)
    h = softmin_keops_line(h, x2, y2, eps)                                          # (B*N1, M2)
    h = h.reshape((B, N1, M2)).permute((0,2,1)).contiguous().reshape((B*M2, 1, N1)) # (B*M2, 1, N1)
    h = softmin_keops_line(h, x2, y2, eps)                                          # (B*M2, M1)
    h = h.reshape((B, M2, M1)).permute((0,2,1)).contiguous()                        # (B, M1, M2)
    return h

def softmin_cuda_image(h, Ms, Ns, eps, dx):
    """
    Compute the online logsumexp of hj - |xi - yj|^2/eps with respect to the variable j, by
    performing "line" logsumexps, where the variables xi and yj are in a 2D grid with spacing dx.
    Dedicated CUDA implementation, faster than the KeOps version for Ms, Ns ≲ 1024.
    """ 
    B = batch_dim(h)
    M1, M2 = Ms
    N1, N2 = Ns
    dx_eff = dx/math.sqrt(eps)
    h = h.view(B*N1, N2).contiguous()                                               # (B*N1, N2)
    h = LogSumExpCUDA(h, M2, dx_eff)                                                 # (B*N1, M2)
    h = h.reshape((B, N1, M2)).permute((0,2,1)).contiguous().reshape((B*M2, N1))    # (B*M2, N1)
    h = LogSumExpCUDA(h, M1, dx_eff)                                                 # (B*M2, M1)
    h = h.reshape((B, M2, M1)).permute((0,2,1)).contiguous()                        # (B, M1, M2)
    return h
