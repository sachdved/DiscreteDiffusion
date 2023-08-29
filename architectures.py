import torch


class FeedForward(torch.nn.Module):
    """
    Simple feedforward layer with an input dim, output dim, and activation
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        activation = torch.nn.ReLU,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer = torch.nn.Linear(input_dim, output_dim)
        self.activation = activation
    def forward(
        self,
        x
    ):
        if self.activation is not None:
            return self.activation(self.layer(x))
        else:
            return self.layer(x)



class AddNorm(torch.nn.Module):
    """
    AddNorm layer, mirroring that of a transformer's architecture
    """
    def __init__(
        self,
        normalized_shape, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_norm = torch.nn.LayerNorm(normalized_shape)
    def forward(
        self,
        x,
        sub_layer_x
    ):
        add = x + sub_layer_x
        return self.layer_norm(add)        



class Attention(torch.nn.Module):
    """
    Dot Product attention class, pretty standard. Nonlinearity is softmax
    """
    def __init__(self,
                 **kwargs
                ):
        super().__init__()


    def forward(queries,
                keys,
                values,
                d_k,
                mask = None
               ):
        scores = torch.matmul(queries, keys.transpose(-1,-2))/torch.sqrt(d_k)
        if mask is not None:
            scores += -1e9*mask

        attention = torch.nn.Softmax(dim=-1)(scores)
        return torch.matmul(attention, values), attention


class MultiHeadedAttention(torch.nn.Module):
    """
    multi-headed attention, pretty standard
    """
    def __init__(self, 
                 heads,
                 d_query,
                 d_key,
                 d_value,
                 d_hidden,
                 d_model,
                 activation = torch.nn.ReLU,
                 attention = DotProductAttention()
                 **kwargs
                ):
        super().__init__(**kwargs)
        self.attention = attention
        self.heads = heads
        self.d_query = d_query
        self.d_key = d_key
        self.d_values = d_values
        self.d_hidden = d_hidden
        self.W_q = torch.nn.Linear(d_query, d_hidden*self.heads)
        self.W_k = torch.nn.Linear(d_key, d_hidden*self.heads)
        self.W_v = torch.nn.Linear(d_values, d_hidden*self.heads)
        self.W_o = torch.nn.Linear(self.heads*d_hidden, d_model)

    def reshape_tensor(self,
                       x,
                       heads,
                       flag
                      ):
        if flag:
            x = x.view(x.shape[0], x.shape[1], heads, x.shape[2]//heads)
            x = x.permute(0,2,1,3)
        else:
            x = x.permute(0,2,1,3)
            x = x.view(x.shape[0], x.shape[1], self.d_model*self.heads)

    def forward(self,
                query,
                keys,
                value
               ):
        q, k, v = self.W_q(query), self.W_k(query), self.W_v(value)
        if self.activation is not None:
            query_reshaped = self.reshape_tensor(self.activation(q), self.heads, True)
            key_reshaped   = self.reshape_tensor(self.activation(k), self.heads, True)
            value_reshaped = self.reshape_tensor(self.activation(v), self.heads, True)

        activations, attention = self.attention(query_reshaped, key_reshaped, value_reshaped)
        activations = self.reshape_tensor(activations, self.heads, False)
        if self.activation is not None:
            return self.W_o(output), output
        else:
            return output, output
        