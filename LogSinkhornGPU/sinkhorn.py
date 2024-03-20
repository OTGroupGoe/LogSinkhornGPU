import torch
from pykeops.torch import LazyTensor
from .aux import *

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
        self.current_error = self.max_error + 1.0
        self.Niter = 0

    # The implementation for getting new duals and computing the error must be
    # written in each subclass
    def get_new_alpha(self):
        """
        Compute and return new alpha. Implementation-dependent.
        """
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of the sinkhorn iteration"
        )

    def get_new_beta(self):
        """
        Compute and return new beta. Implementation-dependent.
        """
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of the sinkhorn iteration"
        )

    def get_cost(self):
        """
        Get cost matrix. Implementation-dependent.
        """
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of cost matrix"
        )

    def get_pi_dense(self):
        """
        Compute dense plan. Implementation-dependent.
        """
        raise NotImplementedError(
            "AbstractSinkhorn has no implementation of dense plan"
        )
    
    def update_beta(self):
        """
        Compute and update beta
        """
        self.beta = self.get_new_beta()

    def update_alpha(self):
        """
        Compute and update alpha
        """
        self.alpha = self.get_new_alpha()

    def get_current_error(self):
        """
        Get L1 Y-marginal error for standard Sinkhorn, without computing
        the full plan. Performs an additional Sinkhorn iter.
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
        for _ in range(niter-1): # last iteration done in `get_current_error`
            self.update_alpha()
            self.update_beta()
        return self.get_current_error()

    def iterate_until_max_error(self):
        """
        Iterate until the sinkhorn error gets below `self.max_error`, or 
        `self.max_iter` iterations are performed.
        """
        max_error = self.max_error
        max_iter = self.max_iter
        if self.max_error_rel:
            max_error *= torch.sum(self.mu)
        while (self.Niter < max_iter) and (self.current_error >= max_error):
            self.current_error = self.iterate(self.inner_iter)
        status = 'converged' if self.current_error < max_error \
            else 'not converged'
        return status

    def change_eps(self, new_eps):
        """
        Change the regularization strength and reset
        error and iteration count
        """
        # NOTE: Careful, offset implementations may need additional steps
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
        """
        Compute and return new alpha
        """
        return - self.eps * (
            softmin_torch(
                (self.beta[:, None, :] - self.C) / self.eps
                + self.lognuref[:, None, :], dim=2
            )
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        return - self.eps * (
            softmin_torch(
                (self.alpha[:, :, None] - self.C) / self.eps
                + self.logmuref[:, :, None], dim=1
            )
            + self.lognuref - self.lognu
        )
    
    def get_cost(self):
        """
        Get cost matrix. May be memory intensive.
        """
        return self.C

    def get_pi_dense(self):
        """
        Compute and return dense plan. May be memory intensive.
        """
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
    nu : torch.Tensor of size (B, N1, N2)
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

        # Get transpose cost matrices
        self.CT = tuple(Ci.permute((0, 2, 1)).contiguous() for Ci in C)
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        h = (self.beta / self.eps + self.lognuref)
        return - self.eps * (
            softmin_torch_image(h, self.C[0], self.C[1], self.eps)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        h = (self.alpha / self.eps + self.logmuref)
        return - self.eps * (
            softmin_torch_image(h, self.CT[0], self.CT[1], self.eps)
            + self.lognuref - self.lognu
        )

    def get_cost(self):
        """
        Get cost matrix.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

    def get_pi_dense(self):
        """
        Compute dense plan.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )


class LogSinkhornKeops(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT, using `pykeops`. Similar 
    implementation to that in `geomloss`, but monitoring the sinkhorn error
    and only stopping when it reaches `self.max_error`.

    Attributes
    ----------
    mu : torch.Tensor of size (B, M)
        First marginals
    nu : torch.Tensor of size (B, N)
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
        """
        Compute and return new alpha
        """
        x, y = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_keops(h, x, y, self.eps) + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        x, y = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_keops(h, y, x, self.eps) + self.lognuref - self.lognu
        )

    def get_cost(self):
        """
        Get cost matrix.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

    def get_pi_dense(self):
        """
        Compute dense plan.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

class LogSinkhornKeopsImage(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, using
    `pykeops`. Similar implementation to that in `geomloss`, but monitoring 
    the sinkhorn error and only stopping when it reaches `self.max_error`.
    Each Sinkhorn iteration has complexity N^(3/2), instead of the usual N^2. 

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
    muref : torch.Tensor with same shape as mu (except axis 0, which can be 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor with same shape as nu (except axis 0, which can be 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        xs, ys = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_keops_image(h, xs, ys, self.eps)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        xs, ys = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_keops_image(h, ys, xs, self.eps)
            + self.lognuref - self.lognu
        )

    def get_cost(self):
        """
        Get cost matrix.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

    def get_pi_dense(self):
        """
        Compute dense plan.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

class LogSinkhornCudaImage(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, 
    custom CUDA implementation. Each Sinkhorn iteration has complexity N^(3/2), 
    instead of the usual N^2. Inspired by the symbolic reduction in `geomloss`,
    but CUDA kernel is optimized for the range of many problems - small size.

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
    muref : torch.Tensor with same shape as mu (except axis 0, which can be 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor with same shape as nu (except axis 0, which can be 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor, or None
        with same dimensions as mu
        Initialization for the first Sinkhorn potential
    """
    def __init__(self, mu, nu, C, eps, **kwargs):
        if isinstance(C, (int, float)):
            dx = C
            dxs = torch.tensor([dx, dx], dtype = mu.dtype)
            dys = dxs
        else:
            xs, ys = C
            
            def get_dx(x):
                """
                Get spacing between points in `x`, checking that grid is 
                1-dimensional, equispaced and starting in zero.
                """
                assert len(x.shape) == 1, "x must be 1-dimensional"
                assert x[0] == 0, "x must start at zero. For a non-zero \
                    offset use `LogSinkhornCudaImageOffset`"
                if len(x) == 1:
                    return 1.0
                else:
                    # Check that gridpoints are equispaced
                    diff = torch.diff(x)
                    dx = diff[0]
                    assert torch.allclose(diff, dx, rtol=1e-4), \
                        "grid points must be equispaced"
                    return dx.item()
                            
            dxs = torch.tensor([get_dx(xi) for xi in xs]).cpu()
            dys = torch.tensor([get_dx(yj) for yj in ys]).cpu()
        Ms = geom_dims(mu)
        Ns = geom_dims(nu)
        assert len(Ms) == len(Ns) == 2, "Shapes incompatible with images"
        super().__init__(mu, nu, (dxs, dys, Ms, Ns), eps, **kwargs)
        # Softmin function assumes inputs of shape (N, dim)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        dxs, dys, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognu
        return - self.eps * (
            softmin_cuda_image(h, Ms, Ns, self.eps, dxs, dys)
            + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        dxs, dys, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmu
        return - self.eps * (
            softmin_cuda_image(h, Ns, Ms, self.eps, dys, dxs)
            + self.lognuref - self.lognu
        )

    def get_cost(self):
        """
        Get cost matrix.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )

    def get_pi_dense(self):
        """
        Compute dense plan.
        """
        raise NotImplementedError(
            "Not implemented yet"
        )
    
class LogSinkhornCudaImageOffset(AbstractSinkhorn):
    """
    Online Sinkhorn solver for standard OT on images with separable cost, 
    custom CUDA implementation. Allows images with offset supports.

    Attributes
    ----------
    mu : torch.Tensor of size (B, M1, M2)
        First marginals
    nu : torch.Tensor of size (B, N1, N2)
        Second marginals 
    C : tuple of the form ((x1, x2), (y1, y2))
        Grid coordinates
    eps : float
        Regularization strength
    muref : torch.Tensor with same shape as mu (except axis 0, which can be 1)
        First reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    nuref : torch.Tensor with same shape as nu (except axis 0, which can be 1)
        Second reference measure for the Gibbs energy, 
        i.e. K = muref \otimes nuref exp(-C/eps)
    alpha_init : torch.Tensor with same shape as mu, or None
        Initialization for the first Sinkhorn potential
    """

    def __init__(self, mu, nu, C, eps, **kwargs):
        xs, ys = C
        B = batch_dim(mu)

        def get_dx(x, B):
            """
            Get spacing between points in `x`, checking that grid is 
            2-dimensional (batch + physical dimension), sharing same 
            batch dimension `B` and equispaced.
            """
            assert len(x.shape) == 2, "x must have a batch and physical dim"
            assert x.shape[0] == B, "x must have `B` as batch dim"
            if x.shape[1] == 1:
                return 1.0
            else:
                # Check that gridpoints are equispaced
                diff = torch.diff(x, dim=1)
                dx = diff[0,0]
                assert torch.allclose(diff, dx, rtol=1e-4), \
                    "grid points must be equispaced"
                return dx.item()

        dxs = torch.tensor([get_dx(xi, B) for xi in xs]).cpu()
        dys = torch.tensor([get_dx(yj, B) for yj in ys]).cpu()

        # Check geometric dimensions
        Ms = geom_dims(mu)
        Ns = geom_dims(nu)
        assert len(Ms) == len(Ns) == 2, "Shapes incompatible with images"

        # Compute the offsets
        self.offsetX, self.offsetY, self.offset_const = \
            compute_offsets_sinkhorn_grid(xs, ys, eps)

        # Save xs and ys in case they are needed later
        self.xs = xs
        self.ys = ys

        C = (dxs, dys, Ms, Ns)

        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        dxs, dys, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognuref + self.offsetY
        return - self.eps * (
            softmin_cuda_image(h, Ms, Ns, self.eps, dxs, dys)
            + self.offsetX + self.offset_const + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        dxs, dys, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmuref + self.offsetX
        return - self.eps * (
            softmin_cuda_image(h, Ns, Ms, self.eps, dys, dxs)
            + self.offsetY + self.offset_const + self.lognuref - self.lognu
        )

    def get_cost(self, ind=None):
        """
        Get dense cost matrix of given problems. If no `ind` is given, all 
        costs are computed. Can be memory intensive, so it is recommended to do 
        small batches at a time.
        """

        if ind == None:
            ind = slice(None,)
        elif isinstance(ind, int):
            ind = [ind]

        xs = tuple(x[ind] for x in self.xs)
        ys = tuple(y[ind] for y in self.ys)
        X = batch_shaped_cartesian_prod(xs)
        Y = batch_shaped_cartesian_prod(ys)
        B = X.shape[0]
        dim = X.shape[-1]
        C = ((X.view(B, -1, 1, dim) - Y.view(B, 1, -1, dim))**2).sum(dim=-1)
        return C, X, Y

    def get_dense_plan(self, ind=None, C=None):
        """
        Get dense plans of given problems. If no argument is given, all plans 
        are computed. Can be memory intensive, so it is recommended to do small 
        batches at a time via the argument `ind`.
        """
        if ind == None:
            ind = slice(None,)
        elif isinstance(ind, int):
            ind = [ind]

        if C == None:
            C, _, _ = self.get_cost(ind)

        B = C.shape[0]
        alpha, beta = self.alpha[ind], self.beta[ind]
        muref, nuref = self.muref[ind], self.nuref[ind]

        pi = torch.exp(
            (alpha.view(B, -1, 1) + beta.view(B, 1, -1) - C) / self.eps
        ) * muref.view(B, -1, 1) * nuref.view(B, 1, -1)
        return pi

    def change_eps(self, new_eps):
        """
        Change the regularization strength `self.eps`.
        In this solver this also involves renormalizing the offsets.
        """
        self.Niter = 0
        self.current_error = self.max_error + 1.
        scale = self.eps / new_eps
        self.offsetX = self.offsetX * scale
        self.offsetY = self.offsetY * scale
        self.offset_const = self.offset_const * scale
        self.eps = new_eps
