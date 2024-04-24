from .aux import *
from .sinkhorn import *
from LogSinkhornGPU.backend import InnerNewtonCUDA

class UnbalancedSinkhornTorch(LogSinkhornTorch): 
    def __init__(self, mu, nu, C, eps, lam, **kwargs):
        self.lam = lam
        self.eps_lam = eps/(1.0 + eps/self.lam)
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        return - self.eps_lam * (
            softmin_torch((self.beta[:,None,:]-self.C)/self.eps + self.lognuref[:,None,:], 2) 
            + 
            self.logmuref - self.logmu
            )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        return - self.eps_lam * (
            softmin_torch((self.alpha[:,:,None]-self.C)/self.eps + self.logmuref[:,:,None], 1) 
            + 
            self.lognuref - self.lognu
            )
    
    def get_current_error(self):
        """
        Get current error for unbalanced Sinkhorn
        """
        new_alpha = self.get_new_alpha()
        # Compute current marginal
        new_mu = self.mu * torch.exp((self.alpha - new_alpha)/self.eps_lam)
        # Update beta (we get an iteration for free)
        self.alpha = new_alpha
        # Finish this sinkhorn iter
        self.update_beta()
        # Return L1 error
        return torch.sum(torch.abs(self.mu - new_mu))

class UnbalancedSinkhornCudaImageOffset(LogSinkhornCudaImageOffset):
    """
    TODO: docstring
    """
    def __init__(self, mu, nu, C, eps, lam, **kwargs):
        self.lam = lam
        self.eps_lam = eps/(1.0 + eps/self.lam)
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        """
        Compute and return new alpha
        """
        dxs, dys, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognuref + self.offsetY
        return - self.eps_lam * (
            softmin_cuda_image(h, Ms, Ns, self.eps, dxs, dys)
            + self.offsetX + self.offset_const + self.logmuref - self.logmu
        )

    def get_new_beta(self):
        """
        Compute and return new beta
        """
        dxs, dys, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmuref + self.offsetX
        return - self.eps_lam * (
            softmin_cuda_image(h, Ns, Ms, self.eps, dys, dxs)
            + self.offsetY + self.offset_const + self.lognuref - self.lognu
        )

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
        self.eps_lam = self.eps/(1.0 + self.eps/self.lam)

    def get_current_error(self):
        """
        Get current error for unbalanced Sinkhorn
        """
        new_alpha = self.get_new_alpha()
        # Compute current marginal
        new_mu = self.mu * torch.exp((self.alpha - new_alpha)/self.eps_lam)
        # Update beta (we get an iteration for free)
        self.alpha = new_alpha
        # Finish this sinkhorn iter
        self.update_beta()
        # Return L1 error
        return torch.sum(torch.abs(self.mu - new_mu))
    
    def get_actual_Y_marginal(self):
        dxs, dys, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmuref + self.offsetX
        scaling = softmin_cuda_image(h, Ns, Ms, self.eps, dys, dxs) \
            + self.offsetY + self.offset_const
        return torch.exp(self.beta / self.eps + scaling) * self.nuref

    def get_actual_X_marginal(self):
        dxs, dys, Ms, Ns = self.C
        h = self.beta / self.eps + self.lognuref + self.offsetY
        scaling = softmin_cuda_image(h, Ms, Ns, self.eps, dxs, dys) \
            + self.offsetX + self.offset_const
        return torch.exp(self.alpha / self.eps + scaling) * self.muref
    
    def primal_score(self):
        PXpi = self.get_actual_X_marginal()
        PYpi = self.get_actual_Y_marginal()

        score = torch.sum(self.alpha * PXpi) + torch.sum(self.beta * PYpi) \
            + self.lam*KL(PXpi, self.mu) + self.lam*KL(PYpi, self.nu)
        return score.item()
    
    def dual_score(self):
        lam = self.lam
        score = - lam * torch.sum((torch.exp(-self.alpha/lam)-1)*self.mu) \
                - lam * torch.sum((torch.exp(-self.beta/lam)-1)*self.nu)
        return score.item()
    
class UnbalancedPartialSinkhornCudaImageOffset(UnbalancedSinkhornCudaImageOffset):
    """
    TODO: docstring
    """
    def __init__(self, mu, nu, C, eps, lam, nu_nJ, newton_iter=10, newton_tol=1e-10,
                 **kwargs):
        dtype = mu.dtype
        self.nu_nJ = nu_nJ
        self.lognu_nJ = log_dens(self.nu_nJ)
        self.newton_iter = newton_iter
        try:
            self.t = kwargs["t"]        
        except KeyError:
            self.t = torch.ones_like(nu)
        # CUDA dispatchment code requires these elemens in tensor form
        self.epst = torch.tensor(eps, dtype = dtype)
        self.lamt = torch.tensor(lam, dtype = dtype)
        self.newton_tolt = torch.tensor(newton_tol, dtype = dtype)
        super().__init__(mu, nu, C, eps, lam, **kwargs)

    # Method `get_new_alpha` stays unchanged.

    def get_new_beta(self):
        dxs, dys, Ms, Ns = self.C
        h = self.alpha / self.eps + self.logmuref + self.offsetX
        logKTu = softmin_cuda_image(h, Ns, Ms, self.eps, dys, dxs) \
            + self.offsetY + self.offset_const + self.lognuref

        # self.t is modified inplace
        InnerNewtonCUDA(
            self.newton_iter, self.newton_tolt, self.epst, self.lamt, 
            self.t, self.lognu,  # TODO lognu or lognuref???
            # self.lognuref,
            self.lognu_nJ, logKTu
        )
        return -self.lam*self.t - self.eps*logKTu
    
    def primal_score(self):
        PXpi = self.get_actual_X_marginal()
        # TODO: Here we need to add them all?
        PYpi = self.get_actual_Y_marginal()

        score = torch.sum(self.alpha * PXpi) + torch.sum(self.beta * PYpi) \
            + self.lam*KL(PXpi, self.mu) + self.lam*KL(PYpi + self.nu_nJ, self.nu)
        return score.item()

    def dual_score(self):
        lam = self.lam
        score = - lam * torch.sum((torch.exp(-self.alpha/lam)-1)*self.mu) \
                - lam * torch.sum((torch.exp(-self.beta/lam)-1)*self.nu) \
                - torch.sum(self.beta, self.nu_nJ)
        return score.item()