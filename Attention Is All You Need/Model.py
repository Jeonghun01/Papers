import torch
import torch.nn as nn
import torch.nn.functional as F


N       = 6   # num_blocks
h       = 8
d_k     = 64
d_v     = 64
d_model = 512
d_ff    = 2048


def Positional_Encoding():
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
        

    def forward(self, x):
        concat_groups = []

        for i, group in enumerate(self.linear_groups):
            Q:torch.Tensor = group[f"linearQ_{i + 1}"](x)
            K:torch.Tensor = group[f"linearK_{i + 1}"](x)
            V:torch.Tensor = group[f"linearV_{i + 1}"](x)

            answer_weight  = torch.matmul(Q, K.permute(0, 2, 1))
            scale          = answer_weight / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))
            softmax        = F.softmax(scale, dim = -1)
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
    def __init__(self, embedding_dict_size):
        super(Encoder, self).__init__()
        # 2 sub-layers
        self.embedding = nn.Embedding(num_embeddings = embedding_dict_size, embedding_dim = d_model)
        # PE
        self.block_groups = nn.ModuleList([
            nn.ModuleDict({
                f"MHA_{i + 1}" : Multi_Head_Attention(),
                f"Norm1_{i + 1}": nn.BatchNorm1d(num_features = d_model),
                f"FF_{i + 1}" : Feed_Forward(),
                f"Norm2_{i + 1}": nn.BatchNorm1d(num_features = d_model),
            })
            for i in range(N)
        ])

    def forward(self, input:torch.Tensor):
        input = input.to(dtype = torch.int32)
        x = self.embedding(input)
        
        for i, block in enumerate(self.block_groups):
            skip_x = x
            x = block[f"MHA_{i + 1}"].forward(x)
            x = x + skip_x
            x = x.permute(0, 2, 1)
            x = block[f"Norm1_{i + 1}"](x)
            x = x.permute(0, 2, 1)

            skip_x = x
            x = block[f"FF_{i + 1}"].forward(x)
            x = x + skip_x
            x = x.permute(0, 2, 1)
            x = block[f"Norm2_{i + 1}"](x)
            x = x.permute(0, 2, 1)
            
        return x

class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        # 3 sub-layers
        pass

    def forward(self,):
        pass