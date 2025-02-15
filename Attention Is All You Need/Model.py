import torch
import torch.nn as nn
import torch.nn.functional as F


N       = 6   # num_blocks
h       = 8
d_k     = 64
d_v     = 64
d_model = 64 * h


def Positional_Encoding():
    pass


class Multi_Head_Attention():
    def __init__(self,):
        super(Multi_Head_Attention, self).__init__()
        self.linear_groups = nn.ModuleList([
            nn.ModuleDict({
                "linearQ": nn.Linear(in_features = d_model, out_features = d_k, bias = False),
                "linearK": nn.Linear(in_features = d_model, out_features = d_k, bias = False),
                "linearV": nn.Linear(in_features = d_model, out_features = d_v, bias = False),
            })
            for i in range(h)
        ])
        self.MHALinear = nn.Linear(in_features = h*d_v, out_features = d_model, bias = False)
        

    def forward(self, x):
        concat_groups = []

        for group in self.linear_groups:
            Q:torch.Tensor = group["linearQ"](x)
            K:torch.Tensor = group["linearK"](x)
            V:torch.Tensor = group["linearV"](x)

            answer_weight = torch.matmul(Q, K.permute(0, 2, 1))
            scale = answer_weight / torch.sqrt(torch.tensor(d_k, dtype = torch.float32))
            softmax = F.softmax(scale, dim = -1)
            attention = torch.matmul(softmax, V)

            concat_groups.append(attention)
        
        concat_result = torch.cat(concat_groups, dim = -1)
        MHA = self.MHALinear(concat_result)

        return MHA







class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        # 2 sub-layers
        pass

    def forward(self,):
        pass


class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        # 3 sub-layers
        pass

    def forward(self,):
        pass