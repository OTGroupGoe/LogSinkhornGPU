import torch
from pykeops.torch import LazyTensor
from LogSinkhornGPU.backend import LogSumExpCUDA
import math

def log_dens(a):
    """
    Log of `a`, thresholded at the negative infinities. Taken from `geomloss`.
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
    Compute the logsumexp of `h` along the dimension `dim`.
    """
    return torch.logsumexp(h, dim=dim, keepdims=False)


def softmin_torch_image(h, C1, C2, eps):
    """
    Compute the logsumexp of `h`, with respect to the cost `C1` along dimension
    1, and `C2` along dimension 2.
    """
    B = batch_dim(h)
    _, M1, N1 = C1.shape  # TODO: use geom_dims
    _, M2, N2 = C2.shape
    # h is assumed to be of shape (B, N1, N2)
    h = h.view(B, N1, 1, N2).contiguous()  # (B, N1, 1, N2)
    h = torch.logsumexp(
        h - C2[:, None, :, :]/eps, dim=3, keepdims=True
    )  # (B, N1, M2, 1)
    h = h.permute((0, 2, 3, 1)).contiguous()  # (B, M2, 1, N1)
    h = torch.logsumexp(
        h - C1[:, None, :, :]/eps, dim=3, keepdims=True
    )  # (B, M2, M1, 1)
    h = h.permute((0, 2, 1, 3)).reshape(B, M1, M2).contiguous()  # (B, M1, M2)
    return h


def softmin_keops(h, x, y, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the 
    variable j. Following `geomloss`.
    """
    B = batch_dim(h)
    xi = LazyTensor(x[:, :, None, :])
    yj = LazyTensor(y[:, None, :, :])
    Cij = ((xi - yj)**2).sum(-1)
    hj = LazyTensor(h[:, None, :, None])
    return (hj-Cij/eps).logsumexp(2).view(B, -1)


def softmin_keops_line(h, x, y, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the 
    variable j. `x` and `y` are unidimensional vectors. 
    Following `geomloss`.
    """
    B = batch_dim(h)
    # If we remove the last dimension keops freaks out
    xi = LazyTensor(x.view(1, -1, 1, 1))
    yj = LazyTensor(y.view(1, 1, -1, 1))
    Cij = (xi - yj)**2
    hj = LazyTensor(h.view(B, 1, -1, 1))
    return (hj-Cij/eps).logsumexp(2).view(B, -1)


def softmin_keops_image(h, xs, ys, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the 
    variable j, by performing "line" logsumexps. 
    Following `geomloss`.
    """
    B = batch_dim(h)
    x1, x2 = xs
    y1, y2 = ys
    M1, M2, N1, N2 = len(x1), len(x2), len(y1), len(y2)
    h = h.reshape(B*N1, 1, N2).contiguous()  # (B*N1, 1, N2)
    h = softmin_keops_line(h, x2, y2, eps)  # (B*N1, M2)
    h = h.reshape((B, N1, M2)).permute((0, 2, 1)).contiguous() \
        .reshape((B*M2, 1, N1))  # (B*M2, 1, N1)
    h = softmin_keops_line(h, x1, y1, eps)  # (B*M2, M1)
    h = h.reshape((B, M2, M1)).permute((0, 2, 1)).contiguous()  # (B, M1, M2)
    return h


def softmin_cuda_image(h, Ms, Ns, eps, dxs, dys = None):
    """
    Compute the online logsumexp of hj - |xi - yj|^2/eps with respect to the 
    variable j, by performing "line" logsumexps, where the variables xi and yj 
    are in respective 2D grid with spacings dxs and dys.
    Dedicated CUDA implementation, faster than the KeOps version for Ms, 
    Ns ≲ 1024.
    """
    if dxs.numel() == 1:
        dxs = dxs * torch.ones(2, dtype = dxs.dtype, device = dxs.device)
    if dys is None: 
        dys = dxs
    B = batch_dim(h)
    M1, M2 = Ms
    N1, N2 = Ns
    dxs_eff = dxs/math.sqrt(eps)
    dys_eff = dys/math.sqrt(eps)
    h = h.view(B*N1, N2).contiguous()  # (B*N1, N2)
    h = LogSumExpCUDA(h, M2, dys_eff[1], dxs_eff[1])  # (B*N1, M2)
    h = h.reshape((B, N1, M2)).permute((0, 2, 1)).contiguous() \
        .reshape((B*M2, N1))    # (B*M2, N1)
    h = LogSumExpCUDA(h, M1, dys_eff[0], dxs_eff[0])  # (B*M2, M1)
    h = h.reshape((B, M2, M1)).permute((0, 2, 1)).contiguous()  # (B, M1, M2)
    return h

def batch_shaped_cartesian_prod(xs):
    """
    Compute the tensor X of shape (B, M1, ..., Md, d) such that
    `X[i] = torch.cartesian_prod(x1[i],...,xd[i]).view(M1, ..., Md, d)`,
    where xs = (x1, ..., xd) is tuple of tensors of shape (B, M1), ... (B, Md).
    """
    B = xs[0].shape[0]
    for x in xs:
        assert B == x.shape[0], "All xs must have the same batch dimension"
        assert len(x.shape) == 2, "xi must have shape (B, Mi)"
    Ms = tuple(x.shape[1] for x in xs)
    dim = len(xs)
    device = xs[0].device

    shapeX = (B, ) + Ms + (dim,)
    X = torch.empty(shapeX, device=device)
    for i in range(dim):
        shapexi = (B,) + (1,)*i + (Ms[i],) + (1,)*(dim-i-1)
        X[..., i] = xs[i].view(shapexi)
    return X

def compute_offsets_sinkhorn_grid(xs, ys, eps):
    """
    Compute offsets for sinkhorn potentials when grid does not start at 0. 
    The entropic OT plan is invariant under constant offsets, because their 
    effect is absorved by the duals. This function computes those dual offsets.

    Arguments
    =========
    xs and ys are d-tuples of tensors with shape (B, Mi) where B is the batch 
    dimension and Mi the size of the grid in that coordinate.
    """
    # Get cartesian prod
    X = batch_shaped_cartesian_prod(xs)
    Y = batch_shaped_cartesian_prod(ys)
    shapeX = X.shape
    B, Ms, dim = shapeX[0], shapeX[1:-1], shapeX[-1]
    Ns = Y.shape[1:-1]

    # Get "bottom left" corner coordinates: select slice (:, 0, ..., 0, :)
    X0 = X[(slice(None),) + (0,)*dim + (slice(None),)] \
        .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.
    Y0 = Y[(slice(None),) + (0,)*dim + (slice(None),)] \
        .view((B,) + (1,)*dim + (dim,))  # NOTE alternatively: use unpack op.

    # Compute the offsets
    offsetX = torch.sum(2*(X-X0)*(Y0-X0), dim=-1)/eps
    offsetY = torch.sum(2*(Y-Y0)*(X0-Y0), dim=-1)/eps
    offset_constant = -torch.sum((X0-Y0)**2, dim=-1)/eps
    return offsetX, offsetY, offset_constant
