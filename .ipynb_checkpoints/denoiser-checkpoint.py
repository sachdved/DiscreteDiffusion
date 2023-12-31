import torch
import numpy as np
import sklearn

from utils import *
from architectures import *
import preprocess

class AttentionDenoiser(torch.nn.Module):
    def __init__(
        self,
        heads,
        d_time,
        d_time_hidden,
        d_seq,
        d_query = 128,
        d_key = 128,
        d_values = 128,
        d_hidden = 128,
        d_model = 128,
        p = 0.1,
        activation = torch.nn.ReLU(),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.heads = heads
        self.activation = activation
        self.forward_time_1 = FeedForward(d_time, d_time_hidden)
        self.forward_time_2 = FeedForward(d_time_hidden, d_time_hidden)
        self.forward_time_3 = FeedForward(d_time_hidden, d_time_hidden)
 
        self.forward_seq_1 = FeedForward(d_seq, d_model)
        self.forward_seq_2 = FeedForward(d_model, d_model)
        self.forward_seq_3 = FeedForward(d_model, d_model)

        self.forward_time_seq = FeedForward(d_model + d_time_hidden, d_model)
        
        self.mha_1 = MultiHeadedAttention(self.heads, d_model, d_model, d_model, d_hidden, d_model)
        self.dropout_1 = torch.nn.Dropout(p)
        self.addnorm_1 = AddNorm(d_model)
        self.feedforward_1 = FeedForward(d_model, d_model)

        self.mha_2 = MultiHeadedAttention(self.heads, d_model, d_model, d_model, d_hidden, d_model)
        self.dropout_2 = torch.nn.Dropout(p)
        self.addnorm_2 = AddNorm(d_model)
        self.feedforward_2 = FeedForward(d_model, d_model)

        self.feedforward_3 = FeedForward(d_model, d_seq)
        
    def forward(self, X, t):

        time_points, batch_size, seq_length, aas = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        
        t = t.reshape(t.shape[0] * t.shape[1], 1)
        X = X.view(time_points*batch_size, seq_length, aas)


        time_encoding = self.forward_time_1(t)
        time_encoding = self.forward_time_2(time_encoding)
        time_encoding = self.forward_time_3(time_encoding)
        time_encoding = time_encoding.view(time_encoding.shape[0], 1, time_encoding.shape[1])

        time_encoding = time_encoding.tile((1, seq_length, 1))
        seq_encoding = self.forward_seq_1(X)
        seq_encoding = self.forward_seq_2(seq_encoding)
        seq_encoding = self.forward_seq_3(seq_encoding)

        
        seq_time_encoding = torch.concat([time_encoding, seq_encoding], dim=-1)

        input_encoding = self.forward_time_seq(seq_time_encoding)

        X_1, _ = self.mha_1(input_encoding, input_encoding, input_encoding)
        X_1    = self.dropout_1(X_1)
        X_1    = self.addnorm_1(X_1, input_encoding)
        X_1    = self.feedforward_1(X_1)

        X_2, _ = self.mha_2(X_1, X_1, X_1)
        X_2    = self.dropout_2(X_2)
        X_2    = self.addnorm_2(X_2,X_1)
        X_2    = self.feedforward_2(X_2)

        X_3    = self.feedforward_3(X_2)
        Y_pred = torch.nn.Softmax(dim=-1)(X_3)
        return Y_pred.view(time_points, batch_size, seq_length, aas)


class MLPDenoiser(torch.nn.Module):
    def __init__(
        self,
        d_time,
        d_seq,
        d_aas,
        d_model = 128,
        p = 0.1,
        activation = torch.nn.ReLU(),
        **kwargs
    ):
        super().__init__(**kwargs)
        self.activation = activation
        self.p = p


        self.forward_time_seq = FeedForward(d_seq*d_aas + d_time, d_model)

        self.forward_time_seq_1 = FeedForward(d_model, d_model)
        self.dropout_1 = torch.nn.Dropout(self.p)
        
        self.forward_time_seq_2 = FeedForward(d_model, d_model)
        self.dropout_2 = torch.nn.Dropout(self.p)
        
        self.forward_time_seq_3 = FeedForward(d_model, d_model)
        self.dropout_3 = torch.nn.Dropout(self.p)

        self.feedforward_final = FeedForward(d_model, d_seq*d_aas)
        
    def forward(self, X, t):

        time_points, batch_size, seq_length, aas = X.shape[0], X.shape[1], X.shape[2], X.shape[3]
        
        t = t.reshape(t.shape[0] * t.shape[1], 1)
        X = X.view(time_points*batch_size, seq_length, aas)
        
        X_flattened = torch.nn.Flatten(start_dim=1)(X)
        
        seq_time_encoding = torch.concat([t, X_flattened], dim=-1)

        input_encoding = self.forward_time_seq(seq_time_encoding)

        X_1 = self.forward_time_seq_1(input_encoding)

        X_1 = self.dropout_1(X_1)

        X_2 = self.forward_time_seq_2(X_1)

        X_2 = self.dropout_2(X_2)

        X_3 = self.forward_time_seq_3(X_2)

        X_3 = self.dropout_3(X_3)

        X_final = self.feedforward_final(X_3)

        X_final = X_final.view(time_points, batch_size, seq_length, aas)
        
        Y_pred = torch.nn.Softmax(dim=-1)(X_final)
        return Y_pred