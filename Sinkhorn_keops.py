from .Aux import batch_dim
from pykeops.torch import LazyTensor

# Only to be imported if one intends to use keops

def softmin_keops(h, x, y, eps):
    """
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the 
    variable j. 
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
    Compute the online logsumexp of hj - |xi-yj|^2/eps with respect to the 
    variable j. `x` and `y` are unidimensional vectors. 
    Inspired by `geomloss`.
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
    Inspired by `geomloss`.
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