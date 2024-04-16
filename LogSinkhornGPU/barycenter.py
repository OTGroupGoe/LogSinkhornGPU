import torch
from .aux import * 
from .sinkhorn import LogSinkhornCudaImage

class BarycenterCudaImage(LogSinkhornCudaImage):
    """
    Barycenter is first marginal
    """
    def __init__(self, nu, xs, eps, **kwargs):
        mu = torch.ones_like(nu) / nu.sum(axis = (1,2), keepdims = True)
        C = (xs, xs)
        # Try to get barycenter weights from kwargs
        try: 
            self.weights = kwargs["weights"]
        except KeyError:
            B = nu.shape[0]
            options = dict(dtype = nu.dtype, device = nu.device)
            self.weights = torch.ones((B, 1, 1), **options) / B
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        dxs, dys, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognuref

        # logsumexp to get scaling factors (use h)
        scaling = softmin_cuda_image(h, Ms, Ns, self.eps, dxs, dys)
        # build new estimate for mu (barycenter)
        new_logmu = torch.sum(
            (self.alpha / self.eps + self.logmuref + scaling) * self.weights,
            dims = 0, keepdims = True
        )
        # compute new potential with standard sinkhorn iteration
        new_alpha = - self.eps * (scaling + self.logmuref - new_logmu)
        return new_alpha, new_logmu
    
    def update_alpha(self):
        new_alpha, new_logmu = self.get_new_alpha()
        self.alpha = new_alpha
        self.logmu = new_logmu.expand_as(new_alpha).contiguous()
        self.mu = torch.exp(self.logmu)
    
    # `get_new_beta` is as in standard sinkhorn
    # `get_current_error` formula works fine here as well.
