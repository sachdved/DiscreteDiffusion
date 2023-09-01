import torch
import numpy as np
import sklearn

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
    results = torch.nn.functional.one_hot(torch.multinomial(probabilities.view(-1, probabilities.shape[-1]), 1), num_classes=num_states)

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
        noised_samples[t,:,:,:] = sampler(noised_samples[t-1,:,:,:], noise_matrix.matrix_power(t), num_states)

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

