import torch
import torch.nn as nn
import torch.nn.functional as F


N       = 6   # num_blocks
h       = 8
d_k     = 64
d_v     = 64
d_model = 64 * h



def Scaled_Dot_Product_Attention(Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor) -> torch.Tensor: # Q, K, V (m, d_model)
    # Can be recatored to Multi Head Attention once a time i think
    
    matmul1 = torch.matmul(Q, K)
    scale = matmul1 / torch.sqrt(K.size(1))
    softmax = F.softmax(scale, dim = -1)
    attention = torch.matmul(softmax, V)

    return attention

class Multi_Head_Attention(): # QW, KW, VW -> (m,d-model) dot (d-model. d_k) -> (m, d_k)
    pass

def Positional_Encoding():
    pass

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