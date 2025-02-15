import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

Nx         = 6 
h          = 8
d_k        = 64
d_v        = 64
d_model    = 512
d_ff       = 2048
vocab_size = 37000


class Positional_Encoding(nn.Module):
    pass


class Multi_Head_Attention(nn.Module):
    def __init__(self,):
        super(Multi_Head_Attention, self).__init__()
        # W_Q, W_K, W_V linear h times
        self.linear_groups = nn.ModuleList([
            nn.ModuleDict({
                f"linearQ_{i + 1}": nn.Linear(in_features = d_model, out_features = d_k, bias = False),
                f"linearK_{i + 1}": nn.Linear(in_features = d_model, out_features = d_k, bias = False),
                f"linearV_{i + 1}": nn.Linear(in_features = d_model, out_features = d_v, bias = False),
            })
            for i in range(h)
        ])
        # W0 linear
        self.MHALinear = nn.Linear(in_features = h*d_v, out_features = d_model, bias = False)
        
    def generate_mask(self, size):
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask == 1 
        return mask

    def apply_mask(self, x, mask):
        mask = mask.unsqueeze(0) 
        mask = mask.expand(x.size(0), -1, -1) 
        masked_attention = x.masked_fill(mask, float('-inf'))
        return masked_attention


    def forward(self, init_Q:torch.Tensor, init_K:torch.Tensor, init_V:torch.Tensor, mask = False):
        if mask == True:
            _, seq_len, _ = init_Q.size()
            masking = self.generate_mask(seq_len)


        concat_groups = []

        for i, group in enumerate(self.linear_groups):
            Q:torch.Tensor = group[f"linearQ_{i + 1}"](init_Q)
            K:torch.Tensor = group[f"linearK_{i + 1}"](init_K)
            V:torch.Tensor = group[f"linearV_{i + 1}"](init_V)

            answer_weight  = torch.matmul(Q, K.permute(0, 2, 1))
            scale          = answer_weight / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))
            softmax        = F.softmax(scale, dim = -1)
            if mask == True:
                softmax = self.apply_mask(softmax, masking)
            attention      = torch.matmul(softmax, V)

           

            concat_groups.append(attention)
        
        concat_result      = torch.cat(concat_groups, dim = -1)
        MHA                = self.MHALinear(concat_result)

        return MHA


class Feed_Forward(nn.Module):
    def __init__(self,):
        super(Feed_Forward, self).__init__()
        self.linear1 = nn.Linear(in_features = d_model, out_features = d_ff, bias = True)
        self.Relu    = nn.ReLU()
        self.linear2 = nn.Linear(in_features = d_ff, out_features = d_model, bias = True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.Relu(x)
        x = self.linear2(x)

        return x


class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.embedding.requires_grad_ = False
        # todo PE

        self.block_groups = nn.ModuleList([
            nn.ModuleDict({
                f"MHA_{i + 1}" : Multi_Head_Attention(),
                f"Norm1_{i + 1}": nn.BatchNorm1d(num_features = d_model),
                f"FF_{i + 1}" : Feed_Forward(),
                f"Norm2_{i + 1}": nn.BatchNorm1d(num_features = d_model),
            })
            for i in range(Nx)
        ])

    def forward(self, input:torch.Tensor):
        input = input.to(dtype = torch.int32)
        x = self.embedding(input)
        x = x * np.sqrt(d_k)
        # todo PE

        for i, block in enumerate(self.block_groups):
            skip_x = x
            x = block[f"MHA_{i + 1}"].forward(init_Q = x, init_K = x, init_V = x, mask = False)
            x = F.dropout(x, p = 0.1)
            x = x + skip_x
            x = x.permute(0, 2, 1)
            x = block[f"Norm1_{i + 1}"](x)
            x = x.permute(0, 2, 1)

            skip_x = x
            x = block[f"FF_{i + 1}"].forward(x)
            x = F.dropout(x, p = 0.1)
            x = x + skip_x
            x = x.permute(0, 2, 1)
            x = block[f"Norm2_{i + 1}"](x)
            x = x.permute(0, 2, 1)
            
        return x



class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = d_model)
        self.embedding.requires_grad_ = False
        # todo PE

        self.block_groups = nn.ModuleList([
            nn.ModuleDict({
                f"Masked_MHA_{i + 1}" : Multi_Head_Attention(),
                f"Norm1_{i + 1}": nn.BatchNorm1d(num_features = d_model),
                f"MHA_{i + 1}" : Multi_Head_Attention(),
                f"Norm2_{i + 1}": nn.BatchNorm1d(num_features = d_model),
                f"FF_{i + 1}" : Feed_Forward(),
                f"Norm3_{i + 1}": nn.BatchNorm1d(num_features = d_model),
            })
            for i in range(Nx)
        ])
        self.linear = nn.Linear(in_features = d_model, out_features = vocab_size)
        self.linear.requires_grad_ = False
        self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, output, encoder_result):
        output = output.to(dtype = torch.int32)
        x = self.embedding(output)
        x = x * np.sqrt(d_k)
        # todo PE

        for i, block in enumerate(self.block_groups):
            skip_x = x
            x = block[f"Masked_MHA_{i + 1}"].forward(init_Q = x, init_K = x, init_V = x, mask = True)
            x = F.dropout(x, p = 0.1)
            x = skip_x + x
            x = x.permute(0, 2, 1)
            x = block[f"Norm1_{i + 1}"](x)
            query = x.permute(0, 2, 1)

            skip_x = query
            x = block[f"MHA_{i + 1}"].forward(init_Q = query, init_K = encoder_result, init_V = encoder_result, mask = False)
            x = F.dropout(x, p = 0.1)
            x = skip_x + x
            x = x.permute(0, 2, 1)
            x = block[f"Norm2_{i + 1}"](x)
            x = x.permute(0, 2, 1)

            skip_x = x
            x = block[f"FF_{i + 1}"].forward(x)
            x = F.dropout(x, p = 0.1)
            x = x + skip_x
            x = x.permute(0, 2, 1)
            x = block[f"Norm3_{i + 1}"](x)
            x = x.permute(0, 2, 1)

        x = self.linear(x)
        x = self.softmax(x)

        return x
