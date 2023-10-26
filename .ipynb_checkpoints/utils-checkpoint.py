import torch
import numpy as np
import sklearn

import numpy as np
import torch
import pandas as pd

from tqdm import tqdm

def Gardner_Energy(vector, patterns, bias, order):
    """
    computes energy of a spin vector in a gardner model, assuming the spins come in as N_samples x N_dim.
    vector: N_samples by N_dims
    patterns: expected to be N_dim by M, number of patterns
    bias: expected to be (N_dim, )
    order: float, probably greater than 0 lol
    
    returns N_samples x 1 vector 
    
    energy = Gardner_Energy(spin_vector, Hopfield_patterns, Hopfield_Bias, order)
    
    """
    
    (N_dim, M) = patterns.shape
    
    bias = bias
    h_term = torch.matmul(vector, bias)
    h_term = h_term.squeeze()
    ##returns N_samples x 1 array, with the bias energies
    
    J_term = torch.sum(abs(torch.matmul(vector, patterns))**order, axis = 1) / (N_dim ** (order-1))
    J_term = J_term.squeeze()
    ##returns a N_samples x M matrix, when summed over axis=1, yields N_samples x 1 matrix
    
    return - h_term - J_term / torch.exp(torch.lgamma(torch.tensor(order+1)))


def Gibbs_Sampler(vector, beta, patterns, bias, order, energy_func):
    """
    performs one step of gibbs sampling on a spin vector data set, assuming spins come as N_samples x N_dim

    vector: N_samples by N_dims
    beta: inverse temperature
    patterns: N_dims by M patterns
    bias: N_dims by 1
    order: order of energy
    energy_func: function, that takes arguments vector, patterns, bias, order, and calculates the negative log likelihood of the data point

    returns:
    resampled_vector: N_samples by N_dim
    
    """

    distribution = torch.distributions.uniform.Uniform(low=0.0, high=1.0)

    original_energy = energy_func(vector, patterns, bias, order)

    
    energies_from_mutating_vector = torch.zeros(vector.shape)
    probability_of_accepting_flip = torch.zeros(vector.shape)
    
    for i in range(vector.shape[1]):
        mutated_vector = vector.clone()
        mutated_vector[:,i] = -1 * mutated_vector[:,i]
        
        energies_from_mutating_vector[:,i] = energy_func(mutated_vector, patterns, bias, order)
    
        probability_of_accepting_flip[:,i] = torch.exp(-beta * (energies_from_mutating_vector[:,i] - original_energy))
    
    mask = 2 * torch.greater(distribution.sample(vector.shape), probability_of_accepting_flip)  -  1
    
    resampled_vector = mask*vector
    
    return resampled_vector

def generate_data(patterns, bias, N_samples = 100, order = 2, beta = 1, gibbs_sampling_rounds = 100, energy_func = Gardner_Energy):
    """
    returns generated data 
    
    input:
        patterns: (N_dim, M) where N_dim is dimension of data, M is number of patterns. Defines the types of data we have
        bias: (N_dim, 1) where N_dim is the dimension of the data. Biases data to look one way over another
        N_samples: int, number of samples
        order: float, greater than 0. Determines the strength of the interactions between the sites
        beta: inverse temperature, float, greater than 0.
        gibbs_sampling_rounds: int, how many rounds of gibbs sampling
        energy_func: python function, energy function

    output:
        spins: N_samples by N_dim, drawn from distribution exp(-beta * (bias * spins + (patterns * spins)^order)/(N_dim^(order-1) * order!))
    """
    
    N_dim = patterns.shape[0]
    M     = patterns.shape[1]

    init_spin_vector = 2 * torch.bernoulli(torch.ones(N_samples, N_dim)*0.5) - 1

    spins = init_spin_vector.clone()
    
    for i in tqdm(range(gibbs_sampling_rounds)):
        spins = Gibbs_Sampler(spins, beta, patterns, bias, order, energy_func)
    return spins

class Noiser():
    def __init__(self,
                 noiser = 'Uniform',
                 beta_t = 0.001,
                 k = 21
    ):
        if noiser == 'Uniform':
            self.noise_matrix = (1.-beta_t) * torch.eye(k)
            self.noise_matrix = self.noise_matrix + beta_t/k * (torch.ones((k,k)) - torch.eye(k))
        elif noiser == 'BERT-LIKE':
            self.noise_matrix = (1.-beta_t) * torch.eye(k+1)
            self.noise_matrix[:,k] = torch.ones(k+1)*beta_t
            self.noise_matrix[k,:] = torch.zeros(k+1)
            self.noise_matrix[k,k] = 1.
        elif noiser == 'Gaussian':
            one_way = torch.arange(0,k,1).unsqueeze(0)
            other_way = torch.arange(0,k,1).unsqueeze(1)

            self.noise_matrix = torch.exp(-4 * (one_way-other_way).pow(2)/((k-1)**2 * beta_t))/torch.sum(torch.exp(-4 * torch.arange(-k+1,k,1)**2/((k-1)**2 * beta_t)))
            diagonal = 1-torch.sum(self.noise_matrix - self.noise_matrix[0,0]*torch.eye(k), dim=1)
            
            mask = torch.diag(torch.ones_like(self.noise_matrix))

            self.noise_matrix = mask * torch.diag(diagonal) + (1.-torch.diag(mask))*self.noise_matrix


def sampler(sequence_matrix, noise_matrix, num_states):
    
    probabilities = torch.matmul(sequence_matrix, noise_matrix.unsqueeze(0))
    results = torch.nn.functional.one_hot(torch.multinomial(probabilities.view(-1, probabilities.shape[-1]).double(), 1), num_classes=num_states)

    results = results.view(probabilities.shape[0], probabilities.shape[1], num_states)

    return results.type(torch.FloatTensor)

def forward_marginal(transition_matrix, forward_step, initial_condition):
    return torch.matmul(initial_condition, transition_matrix.matrix_power(forward_step))

def reverse_marginal(transition_matrix, forward_step, reverse_step, initial_condition, present_condition):
    numer = torch.matmul(present_condition, transition_matrix.t()) * torch.matmul(initial_condition, transition_matrix.matrix_power(forward_step-1))
    denom = torch.matmul(torch.matmul(initial_condition, transition_matrix.matrix_power(forward_step)), present_condition.t())

    denom = torch.diagonal(denom, dim1=-2, dim2=-1)
    return numer/denom

def noiser(sequence_matrix, noise_matrix, number_of_noising_steps, num_states):
    
    ts = torch.arange(0,number_of_noising_steps).tile((sequence_matrix.shape[0], 1))
    noised_samples = torch.zeros((number_of_noising_steps, sequence_matrix.shape[0], sequence_matrix.shape[1], sequence_matrix.shape[2]))
    noised_samples[0,:,:,:] = sequence_matrix
    
    for t in range(1, number_of_noising_steps):
        noised_samples[t,:,:,:] = sampler(noised_samples[t-1,:,:,:], noise_matrix, num_states)

    return ts.permute(-1,-2).type(torch.FloatTensor), noised_samples


class ProteinDataset(torch.utils.data.Dataset):
    """
    takes in sequence data and phenotype data and spits back two dictionaries, X with one key - sequence, and Y with one, potentially two, keys - sequence and phenotype
    inputs:
        seq_data: np.array representing sequences
        phenotype_data np.array representing phenotypes
    
    """
    
    def __init__(self,
                 seq_data,
                 phenotype_data = None,
                 include_mask = False,
                **kwargs):
        super().__init__(**kwargs)
        self.include_mask = include_mask
        self.seqs = seq_data
        self.phenotype_data = phenotype_data
        self.AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV-"
        self.IDX_TO_AA = list(self.AMINO_ACIDS)
        self.AA_TO_IDX = {aa: i for i, aa in enumerate(self.IDX_TO_AA)}

    def __len__(self):
        return self.seqs.shape[0]

    def __getitem__(self, index):
        X = dict()
        Y = dict()
        if self.phenotype_data is not None:
            Y['pheno'] = self.phenotype_data[index]

        one_hot_seq = self._to_one_hot(self.seqs[index])
        Y['seq'] = one_hot_seq
        X['seq'] = one_hot_seq
        return X, Y

    def _to_one_hot(self, seq):
        if self.include_mask:
            one_hot_encoded = np.zeros((seq.shape[0],len(self.IDX_TO_AA)+1))
            for index, char in enumerate(seq):
                one_hot_encoded[index, self.AA_TO_IDX[char]]=1
            return torch.tensor(one_hot_encoded, dtype=torch.float32)
        else:
            one_hot_encoded = np.zeros((seq.shape[0],len(self.IDX_TO_AA)))
            for index, char in enumerate(seq):
                one_hot_encoded[index, self.AA_TO_IDX[char]]=1
            return torch.tensor(one_hot_encoded, dtype=torch.float32)

