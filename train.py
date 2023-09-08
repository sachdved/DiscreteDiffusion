import torch
import numpy as np
import sklearn

import pandas as pd

def main(epochs, dataset, lr, cat_diff, batch_size):

    cat_diff.train()
    optim = torch.optim.Adam(cat_diff.parameters(), lr = lr)
    
    for epoch in range(epochs):
        overall_loss = 0
        spins_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch in spins_loader:
            X, Y = batch
            
            X_spins = X['spins']
            Y_spins = Y['spins']
            
            ts, noised_samples = noiser(X_spins, noise_matrix, 100, X_spins.shape[-1])
    
            cat_diff.decode(noised_samples, ts)
            cat_diff.calc_forward_conditionals(noised_samples)

            Lt0_t1_loss = cat_diff.L_t0t1(Y_spins)
            Ltminus1    = cat_diff.L_tminus1(Y_spins, noised_samples)
    
            loss = Lt0_t1_loss + Ltminus1
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            overall_loss += loss.item()
        print('overall loss at epoch {} is '.format(epoch) + str(overall_loss/X_train.shape[0]))

    return cat_diff