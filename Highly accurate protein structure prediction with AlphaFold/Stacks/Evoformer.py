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
        self.output = nn.Linear(in_features = c * N_head, out_features = c_m)

    def forward(self, msa, pair, c = 32):
        output_list = []

        for i, group in enumerate(self.stacks_group):
            msa  = group['norm_msa'](msa)
            q    = group['qkv'](msa)  
            k    = group['qkv'](msa)
            v    = group['qkv'](msa)
            pair = pair.view(-1, c_z)
            b    = group['bias'](group['norm_pair'](pair))
            b    = b.view(s, r, r)
            g    = group['sigmoid'](group['linear_g'](msa))
            
            plus_bias = (1/np.sqrt(c)) * (torch.matmul(q, k.transpose(1, 2)) + b)
            a    = group['softmax'](plus_bias)
            o    = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output = torch.cat(output_list, dim = 2)
        msa_ = self.output(output)
        
        return msa_




class MSAColumnAttention(nn.Module):
    def __init__(self, c = 32, N_head = 8):
        super(MSAColumnAttention, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm_msa'  : nn.LayerNorm(normalized_shape = c_m),
                'qkv'       : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_g'  : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'sigmoid'   : nn.Sigmoid(),
                'softmax'   : nn.Softmax(dim = -2)
            })
            for i in range(N_head)
        ])
        self.output = nn.Linear(in_features = c * N_head, out_features = c_m)

    def forward(self, msa, c = 32):
        output_list = []
        msa = msa.permute(1, 0, 2)

        for i, group in enumerate(self.stacks_group):
            msa = group['norm_msa'](msa)
            q   = group['qkv'](msa)  
            k   = group['qkv'](msa)
            v   = group['qkv'](msa)
            g   = group['sigmoid'](group['linear_g'](msa))
            
            in_softmax = (1/np.sqrt(c)) * (torch.matmul(q, k.transpose(1,2)))
            a   = group['softmax'](in_softmax)
            o   = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output  = torch.cat(output_list, dim = 2)
        msa_    = self.output(output).permute(1, 0, 2)
        
        return msa_ 




class MSATranslation(nn.Module):
    def __init__(self, n = 4):
        super(MSATranslation, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape = c_m)
        self.linear1 = nn.Linear(in_features = c_m, out_features = n * c_m)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features = n * c_m, out_features = c_m)

    def forward(self, msa):
        msa = self.norm(msa)
        msa = self.linear1(msa)
        msa = self.relu(msa)
        msa_ = self.linear2(msa)

        return msa_




class OuterProductMean(nn.Module):
    def __init__(self, c = 32):
        super(OuterProductMean, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape = c_m)
        self.linear_ab = nn.Linear(in_features = c_m, out_features = c)
        self.linear = nn.Linear(in_features = c ** 2, out_features = c_z)

    def forward(self, msa):
        msa = msa.permute(1, 0, 2)
        msa = self.norm(msa)
        a = self.linear_ab(msa)
        b = self.linear_ab(msa) 
        o = torch.flatten(input = torch.mean(input = torch.einsum('rsx,rsy->rsxy',a, b), dim = 1), start_dim = 1, end_dim = 2)
        pre_pair = self.linear(o)
        pair_ = pre_pair.unsqueeze(1).expand(r, r, c_z)
        
        return pair_




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