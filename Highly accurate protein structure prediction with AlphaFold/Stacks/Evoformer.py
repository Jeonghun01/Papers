import torch
import torch.nn as nn

import numpy as np

s = None
r = None
c_m = 256
c_z = 128


class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c = 32, N_head = 8):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm_msa'  : nn.LayerNorm(normalized_shape = c_m),
                'qkv'       : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'norm_pair' : nn.LayerNorm(normalized_shape = c_z),
                'bias'      : nn.Linear(in_features = c_z, out_features = s, bias = False),
                'linear_g'  : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'sigmoid'   : nn.Sigmoid(),
                'softmax'   : nn.Softmax(dim = -2)
            })
            for i in range(N_head)
        ])
        self.output    = nn.Linear(in_features = c * N_head, out_features = c_m)

    def forward(self, msa, pair, c = 32, N_head = 8):
        output_list = []

        for i, group in enumerate(self.stacks_group):
            msa = group['norm_msa'](msa)
            q = group['qkv'](msa)  
            k = group['qkv'](msa)
            v = group['qkv'](msa)
            pair = pair.view(-1, c_z)
            b = group['bias'](group['norm_pair'](pair))
            b = b.view(s, r, r)
            g = group['sigmoid'](group['linear_g'](msa))
            
            plus_bias = (1/np.sqrt(c)) * (torch.matmul(q, k.transpose(1, 2)) + b)
            a = group['softmax'](plus_bias)
            o = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output = torch.cat(output_list, dim = 2)
        msa_ = self.output(output)
        
        return msa_


class MSA_column_wise_gated_self_attention(nn.Moudule):
    def __init__(self,):
        super(MSA_column_wise_gated_self_attention, self).__init__()
        pass

    def forward(self,):
        pass


class MSA_translation(nn.Module):
    def __init__(self,):
        super(MSA_translation, self).__init__()
        pass

    def forward(self,):
        pass


class Outer_product_mean(nn.Module):
    def __init__(self,):
        super(Outer_product_mean, self).__init__()
        pass

    def forward(self,):
        pass


class Triangular_multiplicative_update(nn.Module):
    def __init__(self,):
        super(Triangular_multiplicative_update, self).__init__()
        pass

    def forward(self,):
        pass


class Triangular_self_attention(nn.Module):
    def __init__(self,):
        super(Triangular_self_attention, self).__init__()
        pass

    def forward(self,):
        pass


class Translation_in_the_pair_stack(nn.Module):
    def __init__(self,):
        super(Translation_in_the_pair_stack, self).__init__()
        pass

    def forward(self,):
        pass