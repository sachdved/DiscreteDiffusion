import torch
import numpy as np
import sklearn

from utils import *
from architectures import *


class CategoricalDiffusion(torch.nn.Module):
    def __init__(
        self,
        denoiser,
        noise_matrix,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.denoiser = denoiser
        self.noise_matrix = noise_matrix
        vals, vecs = torch.linalg.eig(self.noise_matrix.t())
        vals = torch.real(vals)
        vecs = torch.real(vecs)
        self.PxT = vecs[:, torch.argmax(vals)].unsqueeze(0).unsqueeze(1)

    def decode(self, noised_samples, ts):
        self.y_pred = self.denoiser(noised_samples, ts)
        
    def L_T(self, noised_sample):        
        dkl_steady_state = torch.sum(
            (noised_sample+1e-6) * torch.log((noised_sample+1e-6)/(self.PxT+1e-6))
        )
    
        return dkl_steady_state

    def recon_loss(self, noised_samples):
        real = noised_samples[0].tile(noised_samples.shape[0]-1, 1, 1, 1)
        return self.cross_entropy(real, self.y_pred[1:])
        
    
    def L_t0t1(self, real):
        t_ones = ts[1].unsqueeze(0)
        y_pred = self.y_pred[1]
        return self.cross_entropy(real, self.y_pred)

    def L_tminus1(self, real, noised_sample):
        
        reverse_marginals = self.one_step_reverse_marginal(real, noised_sample)
        denoised = self.y_pred
        px_tminus1_giv_xt = self.px_tminus1_giv_xt(noised_samples) 

        return torch.sum(reverse_marginals * torch.log((reverse_marginals+1e-6)/(px_tminus1_giv_xt+1e-6)))

    def cross_entropy(self, real, y_pred):
        return -torch.sum(real * torch.log(y_pred+1e-6))

    def one_step_reverse_marginal(self, real, noised_sample):
        reverse_marginals = torch.zeros(noised_samples.shape[0]-2, noised_samples.shape[1], noised_samples.shape[2], noised_samples.shape[3])
        x0=real
        for t in range(2, noised_samples.shape[0]):
            
            xt = noised_samples[t]
            numer = torch.matmul(xt, noise_matrix.t()) * torch.matmul(x0, self.noise_matrix.matrix_power(t-1))
            denom = torch.matmul(torch.matmul(x0, self.noise_matrix.matrix_power(t)), xt.permute(0,2,1))
            denom = torch.diagonal(denom, dim1=-2, dim2=-1).unsqueeze(-1)
            reverse_marginals[t-2] = numer/denom
            
        return reverse_marginals

    def px_tminus1_giv_xt(self, noised_samples):
        denoised = self.y_pred
        px_onestepback = torch.zeros(denoised.shape[0]-2, denoised.shape[1], denoised.shape[2], denoised.shape[3])
        
        for t in range(2, denoised.shape[0]):
            
            denoised_estimate = denoised[t]
            real = noised_samples[0]
            weighted_expectation = real*denoised_estimate

            px_onestepback[t-2] = torch.matmul(weighted_expectation, self.noise_matrix.matrix_power(t-1))
        return px_onestepback
    