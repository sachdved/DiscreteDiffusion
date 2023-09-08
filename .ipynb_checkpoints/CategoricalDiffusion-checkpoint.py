import torch
import numpy as np
import sklearn

from utils import *
from architectures import *


class CategoricalDiffusion(torch.nn.Module):
    """
    Base class for Categorical diffusion.
    input:
        noise_matrix: tensor, K x K states
        denoiser: torch.nn.Module, outputs denoised samples
    """
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
        """
        Forward pass of denoiser
        input:
            noised_samples: tensor (num_time_steps, batch_size, seq_length, num_classes)
            ts: tensor (num_time_steps, batch_size)
        """
        self.y_pred = self.denoiser(noised_samples, ts)

    def calc_forward_conditionals(self, noised_samples):
        """
        computes forward conditionals of the noised samples
        input:
            noised_samples: tensor (num_time_steps, batch_size, seq_length, num_classes)
        """
        forward_conditionals = torch.zeros(noised_samples.shape)

        for t in range(forward_conditionals.shape[0]):
            forward_conditionals[t] = torch.matmul(noised_samples[0], self.noise_matrix.matrix_power(t))
            
        self.forward_conditionals = forward_conditionals

    def one_step_reverse_conditional(self, real, noised_sample):
        """
        computes the one step reverse conditional
        input:
            real: tensor, (batch_size, seq_length, num_states)
            noised_samples: tensor, (num_time_steps, batch_size, seq_length, num_states)
        output:
            reverse_conditionals: tensor, (num_time_steps-2, batch_size, seq_length, num_states)
        """
        reverse_conditionals = torch.zeros(
            noised_samples.shape[0]-2, noised_samples.shape[1], noised_samples.shape[2], noised_samples.shape[3]
        )
        x0=real
        for t in range(2, noised_samples.shape[0]):
            
            xt = noised_samples[t]
            numer = torch.matmul(xt, noise_matrix.t()) * torch.matmul(x0, self.noise_matrix.matrix_power(t-1))
            denom = torch.matmul(torch.matmul(x0, self.noise_matrix.matrix_power(t)), xt.permute(0,2,1))
            denom = torch.diagonal(denom, dim1=-2, dim2=-1).unsqueeze(-1)
            reverse_conditionals[t-2] = numer/denom
            
        return reverse_conditionals

    def q_xtminus1_xt_giv_x0(self, noised_samples, reverse_conditionals):
        
        q_xtminus1_xt_giv_x0 = torch.zeros(
            (noised_samples.shape[0]-2, noised_samples.shape[1], noised_samples.shape[2], noised_samples.shape[3])
        )

        for t in range(q_xtminus1_xt_giv_x0.shape[0]):
            q_xtminus1_xt_giv_x0[t] = reverse_conditionals[t] * torch.matmul(noised_samples[0], self.noise_matrix.matrix_power(t+2))

        return q_xtminus1_xt_giv_x0

    def L_T(self, noised_sample): 
        """
        computes D_KL of noised samples from the steady state distribution. Not relevant for gradient updates
        input:
            noised_samples: tensor (num_time_steps, batch_size, seq_length, num_classes)
        output:
            dkl_steady_state: tensor, (1, grad=None)
        """
        dkl_steady_state = torch.mean(
            torch.sum(
                (noised_sample+1e-6) * torch.log((noised_sample+1e-6)/(self.PxT+1e-6)),
                axis=-1
            )
        )
    
        return dkl_steady_state        
    
    def L_t0t1(self, real):
        """
        computes cross entropy of the one step decoder. requires self.decode and self.calc_forward_conditionals
        input:
            real: tensor, (batch_size, seq_length, states)
        output:
            one_step_cross_ent_loss: tensor, (1, grad=True)
        """
        y_pred = self.y_pred[1]
        return -torch.mean(torch.sum(self.forward_conditionals[1] * torch.log(y_pred),axis=-1))

        

    def L_tminus1(self, real, noised_sample):
        """
        computes D_KL of each one step backwards decoding
        input:
            real: tensor, (batch_size, seq_length, states)
            noised_sample: tensor, (num_time_steps, batch_size, seq_length, states)
        output:
            d_kl_per_time_step: tensor, (1, grad=True)
        """
        
        reverse_conditionals = self.one_step_reverse_conditional(real, noised_sample)

        
        q_xtminus1_xt_giv_x0 = self.q_xtminus1_xt_giv_x0(noised_sample, reverse_conditionals)

        denoised = self.y_pred[2:]

        px_tminus1_giv_xt = q_xtminus1_xt_giv_x0 * denoised
        
        return torch.mean(
            torch.sum(
                self.forward_conditionals[2:] * (
                    reverse_conditionals * torch.log(
                        (reverse_conditionals + 1e-6)/(px_tminus1_giv_xt+1e-6)
                    )
                ),
                axis=-1
            )
        )