import torch
from LogSinkhornGPU import *
import matplotlib.pyplot as plt


"""
Solve the problem 

eps*KL(pi | K) + eta*KL(P_X pi | mu) + eta*KL(P_Y pi | nu)

Sinkhorn iteration is given by 
"""
class UnbalancedSinkhornTorch(AbstractSinkhorn):
    def __init__(self, mu, nu, C, eps, **kwargs):
        self.eta = kwargs["eta"]
        self.eps_eta = eps/(1.0 + eps/self.eta)
        super().__init__(mu, nu, C, eps, **kwargs)

    def get_new_alpha(self):
        return - self.eps_eta * (
            softmin_torch((self.beta[:,None,:]-self.C)/self.eps + self.lognuref[:,None,:], 2) 
            + 
            self.logmuref - self.logmu
            )

    def get_new_beta(self):
        return - self.eps_eta * (
            softmin_torch((self.alpha[:,:,None]-self.C)/self.eps + self.logmuref[:,:,None], 1) 
            + 
            self.lognuref - self.lognu
            )

    def get_pi_dense(self):
        return torch.exp((self.alpha[:,:,None] + self.beta[:,None,:] - self.C)/self.eps + self.logmuref[:,:,None] + self.lognuref[:,None,:])

    def get_current_error(self):
        """
        Get current error for unbalanced Sinkhorn
        """
        new_alpha = self.get_new_alpha()
        # Compute current marginal
        new_mu = self.mu * torch.exp((self.alpha - new_alpha)/self.eps_eta)
        # Update beta (we get an iteration for free)
        self.alpha = new_alpha
        # Finish this sinkhorn iter
        self.update_beta()
        # Return L1 error
        return torch.sum(torch.abs(self.mu - new_mu))