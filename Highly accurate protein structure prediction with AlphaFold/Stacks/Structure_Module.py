import torch
import torch.nn as nn


s           = 50  # random
r           = 100 # random
c_m         = 256 # on paper
c_z         = 128 # on paper


"""
       
"""



# todo - solve point dimesion problem
class InvariantPointAttention(nn.Module):
    def __init__(self, N_head = 12, c= 16, N_q = 4, N_v = 8):
        super(InvariantPointAttention, self).__init__()
        pass

    def forward(self, msa_s, pair, T):
        pass




class BackboneUpdate(nn.Module):
    def __init__(self,):
        super(BackboneUpdate, self).__init__()
        self.linear_b = nn.Linear(in_features = c_m, out_features = 1)
        self.linear_c = nn.Linear(in_features = c_m, out_features = 1)
        self.linear_d = nn.Linear(in_features = c_m, out_features = 1)
        self.linear_t = nn.Linear(in_features = c_m, out_features = 3)

    def forward(self, msa_s):
        b = self.linear_b(msa_s)
        c = self.linear_c(msa_s)
        d = self.linear_d(msa_s)
        t = self.linear_t(msa_s)

        a = 1 / torch.sqrt(1 + torch.multiply(b, b) + torch.multiply(c, c) + torch.multiply(d, d))
        b = b / torch.sqrt(1 + torch.multiply(b, b) + torch.multiply(c, c) + torch.multiply(d, d))
        c = c / torch.sqrt(1 + torch.multiply(b, b) + torch.multiply(c, c) + torch.multiply(d, d))
        d = d / torch.sqrt(1 + torch.multiply(b, b) + torch.multiply(c, c) + torch.multiply(d, d))

        R = torch.stack([
                        torch.stack([
                            torch.multiply(a, a) + torch.multiply(b, b) - torch.multiply(c, c) - torch.multiply(d, d), 
                            2 * torch.multiply(b, c) - 2 * torch.multiply(a, d),
                            2 * torch.multiply(b, d) + 2 * torch.multiply(a, c)
                         ]),
                         torch.stack([
                            2 * torch.multiply(b, c) + 2 * torch.multiply(a, d),
                            torch.multiply(a, a) - torch.multiply(b, b) + torch.multiply(c, c) - torch.multiply(d, d),
                            2 * torch.multiply(c, d) - 2 * torch.multiply(a, b)
                         ]),
                         torch.stack([
                            2 * torch.multiply(b, d) - 2 * torch.multiply(a, c),
                            2 * torch.multiply(c, d) + 2 * torch.multiply(a, b),
                            torch.multiply(a, a) - torch.multiply(b, b) - torch.multiply(c, c) + torch.multiply(d, d)
                         ])
                        ])
        T_ = (R, t)

        return T_




# Todo - When make full model
class computeAllAtomCoordinates(nn.Module):
    def __init__(self,):
        super(computeAllAtomCoordinates, self).__init__()

    def forward(self,):
        pass


class renameSymmetricGroundTruthAtoms(nn.Module):
    def __init__(self,):
        super(renameSymmetricGroundTruthAtoms, self).__init__()

    def forward(self,):
        pass
