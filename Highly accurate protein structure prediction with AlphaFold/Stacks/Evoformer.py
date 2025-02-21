import torch
import torch.nn as nn

import numpy as np




s           = 50  # random
r           = 100 # random
c_m         = 256 # on paper
c_z         = 128 # on paper

"""
        s      = sequence (protein)
        r      = #amino acids
        c_m    = msa feature 
        c_z    = pair feature
        c      = layer units
        N_head = #self-attention
"""




class MSARowAttentionWithPairBias(nn.Module):
    def __init__(self, c = 32, N_head = 8):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm_msa'    : nn.LayerNorm(normalized_shape = c_m),
                'linear_q'    : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_k'    : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_v'    : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'norm_pair'   : nn.LayerNorm(normalized_shape = c_z),
                'linear_b'    : nn.Linear(in_features = c_z, out_features = s, bias = False),
                'linear_g'    : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'sigmoid_g'   : nn.Sigmoid(),
                'softmax_a'   : nn.Softmax(dim = -2)
            })
            for i in range(N_head)
        ])
        self.output = nn.Linear(in_features = c * N_head, out_features = c_m)

    def forward(self, msa, pair, c = 32):
        output_list = []

        for i, group in enumerate(self.stacks_group):
            p_msa  = group['norm_msa'](msa)
            q      = group['linear_q'](p_msa)  
            k      = group['linear_k'](p_msa)
            v      = group['linear_v'](p_msa)
            p_pair = pair.view(-1, c_z)
            b      = group['linear_b'](group['norm_pair'](p_pair))
            b      = b.view(s, r, r)
            g      = group['sigmoid_g'](group['linear_g'](p_msa))
            
            plus_bias = (1/np.sqrt(c)) * (torch.matmul(q, k.transpose(1, 2)) + b)
            a      = group['softmax_a'](plus_bias)
            o      = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output = torch.cat(output_list, dim = 2)
        msa_   = self.output(output)
        
        return msa_




class MSAColumnAttention(nn.Module):
    def __init__(self, c = 32, N_head = 8):
        super(MSAColumnAttention, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm_msa'   : nn.LayerNorm(normalized_shape = c_m),
                'linear_q'   : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_k'   : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_v'   : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'linear_g'   : nn.Linear(in_features = c_m, out_features = c, bias = False),
                'sigmoid_g'  : nn.Sigmoid(),
                'softmax_a'  : nn.Softmax(dim = -2)
            })
            for i in range(N_head)
        ])
        self.output = nn.Linear(in_features = c * N_head, out_features = c_m)

    def forward(self, msa, c = 32):
        output_list = []
        msa = msa.permute(1, 0, 2)

        for i, group in enumerate(self.stacks_group):
            p_msa = group['norm_msa'](msa)
            q     = group['linear_q'](p_msa)  
            k     = group['linear_k'](p_msa)
            v     = group['linear_v'](p_msa)
            g     = group['sigmoid_g'](group['linear_g'](p_msa))
            
            input = (1/np.sqrt(c)) * (torch.matmul(q, k.transpose(1,2)))
            a     = group['softmax_a'](input)
            o     = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output  = torch.cat(output_list, dim = 2)
        msa_    = self.output(output).permute(1, 0, 2)
        
        return msa_ 




class MSATranslation(nn.Module):
    def __init__(self, n = 4):
        super(MSATranslation, self).__init__()
        self.norm    = nn.LayerNorm(normalized_shape = c_m)
        self.linear1 = nn.Linear(in_features = c_m, out_features = n * c_m)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(in_features = n * c_m, out_features = c_m)

    def forward(self, msa):
        msa  = self.norm(msa)
        msa  = self.linear1(msa)
        msa  = self.relu(msa)
        msa_ = self.linear2(msa)

        return msa_




class OuterProductMean(nn.Module):
    def __init__(self, c = 32):
        super(OuterProductMean, self).__init__()
        self.norm     = nn.LayerNorm(normalized_shape = c_m)
        self.linear_a = nn.Linear(in_features = c_m, out_features = c)
        self.linear_b = nn.Linear(in_features = c_m, out_features = c)
        self.linear   = nn.Linear(in_features = c ** 2, out_features = c_z)

    def forward(self, msa):
        msa = msa.permute(1, 0, 2)
        msa = self.norm(msa)
        a   = self.linear_a(msa)
        b   = self.linear_b(msa) 
        o   = torch.flatten(input = torch.mean(input = torch.einsum('rsx,rsy->rsxy',a, b), dim = 1), start_dim = 1, end_dim = 2)
        pre_pair = self.linear(o)
        pair_ = pre_pair.unsqueeze(1).expand(r, r, c_z)
        
        return pair_




class TriangularMultiplicationOutgoing(nn.Module):
    def __init__(self, c = 128):
        super(TriangularMultiplicationOutgoing, self).__init__()
        self.norm1            = nn.LayerNorm(normalized_shape = c_z)
        self.linear_a         = nn.Linear(in_features = c_z, out_features = c)
        self.linear_b         = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_a = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_b = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_g = nn.Linear(in_features = c_z, out_features = c_z)
        self.sigmoid_a        = nn.Sigmoid()
        self.sigmoid_b        = nn.Sigmoid()
        self.sigmoid_g        = nn.Sigmoid()
        self.norm2            = nn.LayerNorm(normalized_shape = c)
        self.linear_out       = nn.Linear(in_features = c, out_features = c_z)

    def forward(self, pair):
        pair  = self.norm1(pair)
        a     = torch.multiply(self.sigmoid_a(self.linear_sigmoid_a(pair)), self.linear_a(pair))
        b     = torch.multiply(self.sigmoid_b(self.linear_sigmoid_a(pair)), self.linear_b(pair))
        g     = self.sigmoid_g(self.linear_sigmoid_g(pair))
        input = torch.sum(torch.multiply(a, b), dim = 1, keepdim = True)
        pair_ = torch.multiply(g, self.linear_out(self.norm2(input)))

        return pair_




class TriangularMultiplicationIncoming(nn.Module):
    def __init__(self, c = 128):
        super(TriangularMultiplicationIncoming, self).__init__()
        self.norm1            = nn.LayerNorm(normalized_shape = c_z)
        self.linear_a         = nn.Linear(in_features = c_z, out_features = c)
        self.linear_b         = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_a = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_b = nn.Linear(in_features = c_z, out_features = c)
        self.linear_sigmoid_g = nn.Linear(in_features = c_z, out_features = c_z)
        self.sigmoid_a        = nn.Sigmoid()
        self.sigmoid_b        = nn.Sigmoid()
        self.sigmoid_g        = nn.Sigmoid()
        self.norm2            = nn.LayerNorm(normalized_shape = c)
        self.linear_out       = nn.Linear(in_features = c, out_features = c_z)

    def forward(self, pair):
        pair  = self.norm1(pair)
        a     = torch.multiply(self.sigmoid_a(self.linear_sigmoid_a(pair)), self.linear_a(pair))
        b     = torch.multiply(self.sigmoid_b(self.linear_sigmoid_a(pair)), self.linear_b(pair))
        g     = self.sigmoid_g(self.linear_sigmoid_g(pair))
        input = torch.sum(torch.multiply(a, b).permute(1, 0, 2), dim = 1, keepdim = True).permute(1, 0, 2)
        pair_ = torch.multiply(g, self.linear_out(self.norm2(input)))

        return pair_





class TriangularAttentionStartingNode(nn.Module):
    def __init__(self, c = 32, N_head = 4):
        super(TriangularAttentionStartingNode, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm'       : nn.LayerNorm(normalized_shape = c_z),
                'linear_q'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_k'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_v'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_b'   : nn.Linear(in_features = c_z, out_features = r, bias = False),
                'linear_g'   : nn.Linear(in_features = c_z, out_features = c),
                'sigmoid_g'  : nn.Sigmoid(),
                'softmax_a'  : nn.Softmax(dim = -1),
            })
            for i in range(N_head)
        ])
        self.linear_out = nn.Linear(in_features = c * N_head, out_features = c_z)

    def forward(self, pair, c = 32):
        output_list = []

        for i, group in enumerate(self.stacks_group):
            p_pair = group['norm'](pair)
            q      = group['linear_q'](p_pair)
            k      = group['linear_k'](p_pair)  
            v      = group['linear_v'](p_pair)         
            b      = group['linear_b'](p_pair)
            g      = group['sigmoid_g'](group['linear_g'](p_pair))

            input = (1 / np.sqrt(c)) * torch.matmul(q, k.transpose(1,2)) + b.transpose(0,1)
            a = group['softmax_a'](input)
            o = torch.multiply(g, torch.matmul(a, v))

            output_list.append(o)

        output = torch.cat(output_list, dim = -1)
        pair_  =  self.linear_out(output)

        return pair_




class TriangularAttentionEndingNode(nn.Module):
    def __init__(self, c = 32, N_head = 4):
        super(TriangularAttentionEndingNode, self).__init__()
        self.stacks_group = nn.ModuleList([
            nn.ModuleDict({
                'norm'       : nn.LayerNorm(normalized_shape = c_z),
                'linear_q'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_k'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_v'   : nn.Linear(in_features = c_z, out_features = c),
                'linear_b'   : nn.Linear(in_features = c_z, out_features = r, bias = False),
                'linear_g'   : nn.Linear(in_features = c_z, out_features = c),
                'sigmoid_g'  : nn.Sigmoid(),
                'softmax_a'  : nn.Softmax(dim = -1),
            })
            for i in range(N_head)
        ])
        self.linear_out = nn.Linear(in_features = c * N_head, out_features = c_z)

    def forward(self, pair, c = 32):
        output_list = []

        for i, group in enumerate(self.stacks_group):
            p_pair = group['norm'](pair)
            q      = group['linear_q'](p_pair)
            k      = group['linear_k'](p_pair)  
            v      = group['linear_v'](p_pair)         
            b      = group['linear_b'](p_pair)
            g      = group['sigmoid_g'](group['linear_g'](p_pair))

            input = ((1 / np.sqrt(c)) * torch.matmul(q, k.permute(1, 0, 2).transpose(1,2)) + b).permute(2, 0, 1)
            a = group['softmax_a'](input)
            o = torch.multiply(g, torch.matmul(a, v.permute(1, 0, 2)).permute(1, 0, 2))

            output_list.append(o)

        output = torch.cat(output_list, dim = -1)
        pair_  =  self.linear_out(output)

        return pair_           




class PairTranslation(nn.Module):
    def __init__(self, n = 4):
        super(PairTranslation, self).__init__()
        self.norm    = nn.LayerNorm(normalized_shape = c_z)
        self.linear1 = nn.Linear(in_features = c_z, out_features = n * c_z)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(in_features = n * c_z, out_features = c_z)

    def forward(self, pair):
        pair  = self.norm(pair)
        pair  = self.linear1(pair)
        pair  = self.relu(pair)
        pair_ = self.linear2(pair)

        return pair_