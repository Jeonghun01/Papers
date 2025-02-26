import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Stacks.Evoformer import MSARowAttentionWithPairBias, MSAColumnAttention, MSATranslation, OuterProductMean
from Stacks.Evoformer import TriangularMultiplicationIncoming, TriangularMultiplicationOutgoing, TriangularAttentionStartingNode, TriangularAttentionEndingNode, PairTranslation

from Stacks.Structure_Module import InvariantPointAttention, Transition, BackboneUpdate, getTorsionAngles
from Loss import torsionAngleLoss, computeFAPE

from Stacks.Structure_Module import computeAllAtomCoordinates


class Evoformer(nn.Module):
    def __init__(self,):
        super(Evoformer, self).__init__()
        self.MSARowAttentionWithPairBias      = MSARowAttentionWithPairBias()
        self.MSAColumnAttention               = MSAColumnAttention()
        self.MSATranslation                   = MSATranslation()
        self.OuterProductMean                 = OuterProductMean()
        self.TriangularMultiplicationIncoming = TriangularMultiplicationIncoming()
        self.TriangularMultiplicationOutgoing = TriangularMultiplicationOutgoing()
        self.TriangularAttentionStartingNode  = TriangularAttentionStartingNode()
        self.TriangularAttentionEndingNode    = TriangularAttentionEndingNode()
        self.PairTranslation                  = PairTranslation()


    def forward(self, msa, pair):
        skip_msa = msa
        msa = self.MSARowAttentionWithPairBias.forward(msa = msa, pair = pair)
        msa = torch.stack([F.dropout(row, p = 0.15) for row in msa])
        msa = skip_msa + msa

        skip_msa = msa
        msa = self.MSAColumnAttention.forward(msa = msa)
        msa = skip_msa + msa

        skip_msa = msa
        msa = self.MSATranslation.forward(msa = msa)
        msa = skip_msa + msa

        pair = self.OuterProductMean.forward(msa = msa)

        skip_pair = pair
        pair = self.TriangularMultiplicationOutgoing.forward(pair = pair)
        pair = torch.stack([F.dropout(row, p = 0.25) for row in pair])
        pair = skip_pair + pair

        skip_pair = pair
        pair = self.TriangularMultiplicationIncoming.forward(pair = pair)
        pair = torch.stack([F.dropout(row, p = 0.25) for row in pair])
        pair = skip_pair + pair

        skip_pair = pair
        pair = self.TriangularAttentionStartingNode.forward(pair = pair)
        pair = torch.stack([F.dropout(row, p = 0.25) for row in pair])
        pair = skip_pair + pair

        skip_pair = pair
        pair = self.TriangularAttentionEndingNode.forward(pair = pair)
        for i in range(pair.size(0)):
            pair[i] = F.dropout(pair[i], p = 0.25)
        pair = skip_pair + pair

        skip_pair = pair
        pair = self.PairTranslation.forward(pair = pair)
        pair = skip_pair + pair

        return msa, pair
    



class Structure_Module(nn.Module):
    def __init__(self,):
        super(Structure_Module, self).__init__()
        self.InvariantPointAttention = InvariantPointAttention()
        self.Transition              = Transition()
        self.BackboneUpdate          = BackboneUpdate()
        self.getTorsionAngles        = getTorsionAngles()
        

    def forward(self, imsa_s, msa_s, pair, T, T_t, x_t, angles_t, angles_at, c = 128):
        #skip_msa_s = msa_s
        #msa_s = InvariantPointAttention(msa_s, pair, T)
        #msa_s = skip_msa_s + msa_s
        #msa_s = F.layer_norm(F.dropout(msa_s, p = 0.1), normalized_shape = c)

        msa_s = self.Transition(msa_s)
        
        T_ = self.BackboneUpdate(msa_s)
        R_T, x_T = T
        R_T_, x_T_ = T_ 
        T = (torch.matmul(R_T, R_T_), torch.matmul(R_T,x_T_.T).transpose(1,2)[:,0,:] + x_T)

        angles = self.getTorsionAngles(msa_s, imsa_s)
        
        R, x = T
        L_aux = computeFAPE(T, x, T_t, x_t, epsilon = 1e-12) + torsionAngleLoss(angles, angles_t, angles_at)

        return L_aux, T, msa_s, angles




class Model(nn.Module):
    def __init__(self, N_evoformer = 48, c_m = 256, c_z = 128, c_s = 128):
        super(Model, self).__init__()
        self.Evoformer_groups = nn.ModuleList([Evoformer() for i in range(N_evoformer)])
        self.Structure_Module = Structure_Module()
        self.norm_pair        = nn.LayerNorm(normalized_shape = c_z)
        self.linear_msa_s     = nn.Linear(in_features = c_m, out_features = c_s)
        self.computAllAtomCoordinates = computeAllAtomCoordinates()

    def forward(self, msa, pair, T_t, x_t, angles_t, angles_at, N_sm = 8, r = 100):

        for evoformer in self.Evoformer_groups:
            msa, pair = evoformer(msa, pair)

        imsa_s = msa[1,:,:]
        pair   = self.norm_pair(pair)
        msa_s  = self.linear_msa_s(imsa_s)
        T = (torch.eye(3).repeat(r, 1, 1), torch.zeros((r,3)))

        L_aux_list = []
        for i in range(N_sm):
            L_aux, T, msa_s, angles = self.Structure_Module(imsa_s, msa_s, pair, T, T_t, x_t, angles_t, angles_at)
            L_aux_list.append(L_aux)
            if i != N_sm - 1:
                T = (torch.eye(3).repeat(r, 1, 1), T[1])
        
        L_aux = np.mean(L_aux_list)
        T_f, x_a = self.computAllAtomCoordinates(T, angles)
        T_f_ = ()
        for Tf in T_f:
            T_n = torch.concat(T, Tf)
            T_f_ += (T_n)
        T_f = T_f_

        