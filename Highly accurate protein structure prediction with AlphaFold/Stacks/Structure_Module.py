import torch
import torch.nn as nn

import types


s           = 50  # random
r           = 100 # random
c_m         = 256 # on paper
c_z         = 128 # on paper




# todo - solve point dimesion problem / will do after study 3D modeling
class InvariantPointAttention(nn.Module):
    def __init__(self, N_head = 12, c= 16, N_q = 4, N_v = 8):
        super(InvariantPointAttention, self).__init__()
    def forward(self, msa_s, pair, T):
        return msa_s




class Transition(nn.Module):
    def __init__(self, c = 128):
        super(Transition, self).__init__()
        self.dropout = nn.Dropout(p = 0.1)
        self.norm    = nn.LayerNorm(normalized_shape = c)
    
    def forward(self, msa_s):
        return self.norm(self.dropout(msa_s))



class BackboneUpdate(nn.Module):
    def __init__(self, c = 128):
        super(BackboneUpdate, self).__init__()
        self.linear_b = nn.Linear(in_features = c, out_features = 1)
        self.linear_c = nn.Linear(in_features = c, out_features = 1)
        self.linear_d = nn.Linear(in_features = c, out_features = 1)
        self.linear_t = nn.Linear(in_features = c, out_features = 3)

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
        R = R.squeeze(dim = -1).permute(2, 0, 1)
        T_ = (R, t)

        return T_




class getTorsionAngles(nn.Module):
    def __init__(self, c = 128, n_torsion = 7):
        super(getTorsionAngles, self).__init__()
        self.torsion_groups = nn.ModuleList([
            nn.ModuleDict({
                'linear_msa_s' : nn.Linear(in_features = c, out_features = c),
                'linear_imsa_s' : nn.Linear(in_features = c_m, out_features = c),
                'linear1'      : nn.Linear(in_features = c, out_features = c),
                'linear2'      : nn.Linear(in_features = c, out_features = c),
                'linear3'      : nn.Linear(in_features = c, out_features = c),
                'linear4'      : nn.Linear(in_features = c, out_features = c),
                'linear_out'    : nn.Linear(in_features = c, out_features = 2),
                'relu1'         : nn.ReLU(),
                'relu2'         : nn.ReLU(),
                'relu3'         : nn.ReLU(),
                'relu4'         : nn.ReLU(),
                'relu_out'      : nn.ReLU()
            })
            for i in range(n_torsion)
        ])
    
    def forward(self, msa_s, imsa_s):
        angles_ = ()

        for i, group in enumerate(self.torsion_groups):
            a = group['linear_msa_s'](msa_s) + group['linear_imsa_s'](imsa_s)
            a = a + group['linear2'](group['relu2'](group['linear1'](group['relu1'](a))))
            a = a + group['linear4'](group['relu4'](group['linear3'](group['relu3'](a))))
            a = group['linear_out'](group['relu_out'](a))
            
            angles_ += (a,)
        
        return angles_




class computeAllAtomCoordinates(nn.Module):
    def __init__(self,):
        super(computeAllAtomCoordinates, self).__init__()
        # Data values on PDB, Alphafold
        self.Transformation_list = [(torch.Tensor([[0.980, 0.150, -0.127],[-0.150, 0.988, 0.040],[0.127, -0.040, 0.991]]), torch.Tensor([0.78, 0.34, 0.26])),
                                    (torch.Tensor([[0.992, -0.110, 0.045],[0.109, 0.994, 0.026],[-0.046, -0.025, 0.998]]), torch.Tensor([0.65, 0.20, -0.05])),
                                    (torch.Tensor([[0.987, 0.143, -0.064],[-0.143, 0.989, 0.029],[0.065, -0.028, 0.997]]), torch.Tensor([0.72, 0.25, -0.08])),
                                    (torch.Tensor([[0.994, 0.108, 0.012],[-0.108, 0.994, -0.032],[-0.012, 0.032, 1.000]]), torch.Tensor([1.30, 0.25, -0.10])),
                                    (torch.Tensor([[0.999, 0.021, -0.034],[-0.021, 0.998, 0.055],[0.034, -0.055, 0.998]]), torch.Tensor([1.10, 0.20, -0.08])),
                                    (torch.Tensor([[0.985, -0.171, 0.040],[0.170, 0.985, 0.043],[-0.041, -0.042, 0.999]]), torch.Tensor([1.24, 0.30, -0.05])),
                                    (torch.Tensor([[0.992, -0.125, 0.011],[0.124, 0.989, -0.073],[-0.020, 0.071, 0.997]]), torch.Tensor([1.05, 0.15, -0.12])) ]
        self.CA_list = [torch.Tensor([1.458, 0.000, 0.000]),
                        torch.Tensor([1.939, 1.524, 0.000]),
                        torch.Tensor([1.939, 2.037, 1.207]),
                        torch.Tensor([2.500, -0.500, -0.500]),
                        torch.Tensor([2.800, 1.000, 0.000]),
                        torch.Tensor([3.200, 1.500, 0.500]),
                        torch.Tensor([3.500, 2.000, 1.000])]
        
        def makeRotX(self, angle):
            R = torch.Tensor([[1, 0, 0],[0, angle[0], -angle[1]],[0, angle[1], angle[0]]])
            t = torch.Tensor([0,0,0])
            T = (R, t)
            return T
        self.makeRotX = types.MethodType(makeRotX, self)

    def forward(self, T, angles):
        angles_n = ()
        for i in range(len(angles)):
            angles_n += (angles[i]/torch.norm(angles[i]),)
        angles = angles_n

        angles_to_T_list = []
        
        for angle in angles:
            angles_to_T_list.append(self.makeRotX(angle[1])) # w, phi, psi, x1, x2, x3, x4
        
        for i in range(7):
            globals()[f"T{i + 1}"] = (torch.matmul(torch.matmul(T[0], self.Transformation_list[i][0]), angles_to_T_list[i][0]),
                                  torch.matmul(torch.matmul(T[0], self.Transformation_list[i][0]), angles_to_T_list[i][1]) + torch.matmul(T[0], self.Transformation_list[i][1]) + T[1])
        
        T_f = (T1, T2, T3, T4, T5, T6, T7)
        for T in T_f:
            pass
        
        #todo get x value, and do rigid transformation, concat

        return T, x










class renameSymmetricGroundTruthAtoms(nn.Module):
    def __init__(self,):
        super(renameSymmetricGroundTruthAtoms, self).__init__()

    def forward(self,):
        pass
