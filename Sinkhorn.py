import torch
from pykeops.torch import LazyTensor
from LogSinkhornGPUBackend import LogSumExpCUDA
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


def softmin_cuda_image(h, Ms, Ns, eps, dx):
    """
    Compute the online logsumexp of hj - |xi - yj|^2/eps with respect to the 
    variable j, by performing "line" logsumexps, where the variables xi and yj 
    are in a 2D grid with spacing dx.
    Dedicated CUDA implementation, faster than the KeOps version for Ms, 
    Ns ≲ 1024.
    """
    B = batch_dim(h)
    M1, M2 = Ms
    N1, N2 = Ns
    dx_eff = dx/math.sqrt(eps)
    h = h.view(B*N1, N2).contiguous()  # (B*N1, N2)
    h = LogSumExpCUDA(h, M2, dx_eff)  # (B*N1, M2)
    h = h.reshape((B, N1, M2)).permute((0, 2, 1)).contiguous() \
        .reshape((B*M2, N1))    # (B*M2, N1)
    h = LogSumExpCUDA(h, M1, dx_eff)  # (B*M2, M1)
    h = h.reshape((B, M2, M1)).permute((0, 2, 1)).contiguous()  # (B, M1, M2)
    return h


class AbstractSinkhorn:
    """
    An abstract class for a Sinkhorn solver. 

    Attributes
    ----------
    mu : torch.Tensor 
        of size (B, M) or (B, M1, ...., Md)
        First marginal (or marginals if B > 1)
    nu : torch.Tensor 
        of size (B, N) or (B, N1, ..., Nd)
        Second marginal (or marginals if B > 1)
    C : general object encoding the cost matrix
        Cost matrix
    eps : float
        Regularization strength
    muref : torch.Tensor 
        of same dimensions as mu 
        (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        of same dimensions as nu 
        (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(
        self, mu, nu, C, eps, muref=None, nuref=None, alpha_init=None,
        inner_iter=20, max_iter=10000, max_error=1e-4,
        max_error_rel=False, get_beta=True, **kwargs
    ):

        self.eps = torch.tensor(eps, dtype=mu.dtype).item()
        self.mu = mu
        self.nu = nu
        self.logmu = log_dens(self.mu)
        self.lognu = log_dens(self.nu)
        self.C = C

        # Batchsize
        self.B = batch_dim(mu)
        assert (batch_dim(mu) == batch_dim(nu)) \
            or (batch_dim(nu) == 1), \
            "mu and nu do not have compatible batch dimensions"

        # Init reference mu
        if muref is not None:
            assert geom_dims(mu) == geom_dims(muref), \
                "mu and muref do not have same geometric dimensions"
            assert (batch_dim(mu) == batch_dim(muref)) \
                or (batch_dim(muref) == 1), \
                "mu and muref do not have compatible batch dimensions"
            self.muref = muref
            self.logmuref = log_dens(muref)
        else:
            self.muref = self.mu
            self.logmuref = self.logmu

        # Init reference nu
        if nuref is not None:
            assert geom_dims(nu) == geom_dims(nuref), \
                "nu and nuref do not have same geometric dimensions"
            assert (batch_dim(nu) == batch_dim(nuref)) \
                or (batch_dim(nuref) == 1), \
                "nu and nuref do not have compatible batch dimensions"
            self.nuref = nuref
            self.lognuref = log_dens(nuref)
        else:
            self.nuref = self.nu
            self.lognuref = self.lognu

        # Initialize alphas if not given
        if alpha_init is not None:
            assert alpha_init.shape == self.mu.shape
            self.alpha = alpha_init
        else:
            self.alpha = torch.zeros_like(self.mu)
        # Perform a first Sinkhorn iteration to initialize beta
        if get_beta:
            self.beta = self.get_new_beta()

        # Error and iteration parameters
        self.max_error = max_error
        self.max_error_rel = max_error_rel
        self.inner_iter = inner_iter
        self.max_iter = max_iter
        self.current_error = self.max_error + 1
        self.Niter = 0

    # The definitions for getting new duals and computing the error must be
    # written in each subclass
    def get_new_alpha(self):
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of the logsumexp"
        )

    def get_new_beta(self):
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of the logsumexp"
        )

    def update_beta(self):
        self.beta = self.get_new_beta()

    def update_alpha(self):
        self.alpha = self.get_new_alpha()

    def get_current_error(self):
        """
        Get current error for standard Sinkhorn
        """
        self.update_alpha()
        new_beta = self.get_new_beta()
        # Compute new marginal
        new_nu = self.nu * torch.exp((self.beta - new_beta)/self.eps)
        # Update beta (we get an iteration for free)
        self.beta = new_beta
        # Return L1 error
        return torch.sum(torch.abs(self.nu - new_nu))

    def iterate(self, niter):
        """
        Iterate a number of times, and compute the current error
        """
        self.Niter += niter
        for _ in range(niter):  # TODO -1 since get_curr_error makes extra iter
            self.update_alpha()
            self.update_beta()
        return self.get_current_error()

    def iterate_until_max_error(self):
        max_error = self.max_error
        if self.max_error_rel:
            max_error *= torch.sum(self.mu)
        while (self.Niter < self.max_iter) and (self.current_error >= max_error):
            self.current_error = self.iterate(self.inner_iter)
        status = 'converged' if self.current_error < self.max_error \
            else 'not converged'
        return status

    def change_eps(self, new_eps):
        """
        Change the regularization strength and reset
        error and iteration count
        """
        # NOTE: Careful, different implementations may need
        # more steps for this
        self.eps = new_eps
        self.Niter = 0
        self.current_error = self.max_error + 1.0   


class LogSinkhornTorch(AbstractSinkhorn):
    """
    A tensorized implementation of a Sinkhorn solver for standard OT 

    Attributes
    ----------
    mu : torch.Tensor of size (B, M)
        First marginals
    nu : torch.Tensor of size (B, N)
        Second marginals 
    C : torch.Tensor of size (1, M, N) or (B, M, N)
        Cost matrix
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor 
        with same dimensions as mu, or None
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        return - self.eps * (
            softmin_torch(
                (self.beta[:, None, :] - self.C) / self.eps
                + self.lognuref[:, None, :], dim=2
            )
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        return - self.eps * (
            softmin_torch(
                (self.alpha[:, :, None] - self.C) / self.eps
                + self.logmuref[:, :, None], dim=1
            )
            + self.lognuref - self.lognu
        )

    def get_pi_dense(self):
        return torch.exp(
            (self.alpha[:, :, None] + self.beta[:, None, :] - self.C)/self.eps
            + self.logmuref[:, :, None] + self.lognuref[:, None, :]
        )


class LogSinkhornTorchImage(AbstractSinkhorn):
    """
    Sinkhorn solver for standard OT on images with separable cost. 
    Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 

    Attributes
    ----------
    mu : torch.Tensor of size (B, M1, M2)
        First marginals
    nu : torch.Tensor 
        of size (B, N1, N2)
        Second marginals 
    C : tuple of torch.Tensor 
        of size (M1, N1) and (M2, N2)
        Cost matrices along each dimension
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):

        # C should be tuple of costs along different dimensions of the measures
        # TODO: check that we are in a 2D problem
        # Check whether the cost has a batch dimension, and if not, add it.
        if len(C[0].shape) == 2:
            for Ci in C:
                assert len(Ci.shape) == 2, \
                    "Dimensions of costs accross dimensions are not consistent"
            C = tuple(Ci[None, :, :] for Ci in C)
        for (i, Ci) in enumerate(C):
            assert Ci.shape[1:] == (mu.shape[i+1], nu.shape[i+1]), \
                "Dimensions of cost and marginal not matching"

        self.CT = tuple(Ci.permute((0, 2, 1)).contiguous() for Ci in C)
        super().__init__(mu, nu, C, eps, **kwargs)
        # Get transpose transport matrix

    def get_new_alpha(self):
        h = (self.beta / self.eps + self.lognuref)
        return - self.eps * (
            softmin_torch_image(h, self.C[0], self.C[1], self.eps)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        h = (self.alpha / self.eps + self.logmuref)
        return - self.eps * (
            softmin_torch_image(h, self.CT[0], self.CT[1], self.eps)
            + self.lognuref - self.lognu
        )

    # def get_pi_dense(self):
    #     return torch.exp(
    #         (self.alpha + self.beta - self.C) / self.eps
    #         + self.logmu + self.lognu
    #     )


class LogSinkhornKeops(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT, using `pykeops`. 

    Attributes
    ----------
    mu : torch.Tensor 
        of size (B, M)
        First marginals
    nu : torch.Tensor 
        of size (B, N)
        Second marginals 
    C : tuple of the form (X, Y)
        Coordinates of mu and nu. Must have shapes of the form 
        (B, M, dim) or (1, M, dim), where dim is the ambient dimension
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        x, y = C
        # Check shapes
        assert len(x.shape) == 3, \
            "x.shape must be of the form (B, M, dim) or (1, M, dim)"
        assert len(y.shape) == 3, \
            "y.shape must be of the form (B, N, dim) or (1, N, dim)"
        # Check consistent dimension
        assert x.shape[-1] == y.shape[-1], \
            "spatial dim of x and y must coincide"
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        x, y = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_keops(h, x, y, self.eps) + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        x, y = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_keops(h, y, x, self.eps) + self.lognuref - self.lognu
        )


class LogSinkhornKeopsImage(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, using
    `pykeops`. 
    Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 
    Inspired greatly on `geomloss`.

    Attributes
    ----------
    mu : torch.Tensor of size (B, M1, M2)
        First marginals
    nu : torch.Tensor of size (B, N1, N2)
        Second marginals 
    C : tuple of the form ((x1, x2), (y1, y2))
        Grid coordinates for the marginals
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        xs, ys = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_keops_image(h, xs, ys, self.eps)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        xs, ys = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_keops_image(h, ys, xs, self.eps)
            + self.lognuref - self.lognu
        )


class LogSinkhornCudaImage(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, custom CUDA implementation. 
    Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 
    Inspired greatly on `geomloss`.

    Attributes
    ----------
    mu : torch.Tensor of size (B, M1, M2)
        First marginals
    nu : torch.Tensor of size (B, N1, N2)
        Second marginals 
    C  : either float or tuple of the form ((x1, x2), (y1, y2))
        Distance between pixels, or grid coordinates
    eps : float
        Regularization strength
    muref : torch.Tensor 
        with same dimensions as mu (except axis 0, which can have len = 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor 
        with same dimensions as nu (except axis 0, which can have len = 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        if isinstance(C, (int, float)):
            dx = C
        else:
            xs, ys = C
            # TODO: check that xs, ys have same dx
            dx = xs[0][1] - xs[0][0]
        Ms = geom_dims(mu)
        Ns = geom_dims(nu)
        assert len(Ms) == len(Ns) == 2, "Shapes incompatible with images"
        super().__init__(mu, nu, (dx, Ms, Ns), eps, **kwargs)
        # Softmin function assumes inputs of shape (N, dim)

    def get_new_alpha(self):
        dx, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_cuda_image(h, Ms, Ns, self.eps, dx)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        dx, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_cuda_image(h, Ns, Ms, self.eps, dx)
            + self.lognuref - self.lognu
        )
