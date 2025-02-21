import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from Stacks.Evoformer import MSARowAttentionWithPairBias, MSAColumnAttention, MSATranslation, OuterProductMean
from Stacks.Evoformer import TriangularMultiplicationIncoming, TriangularMultiplicationOutgoing, TriangularAttentionStartingNode, TriangularAttentionEndingNode, PairTranslation

class Evoformer(nn.Module):
    def __init__(self,):
        super(Evoformer, self).__init__()
        self.MSARowAttentionWithPairBias = MSARowAttentionWithPairBias()
        self.MSAColumnAttention = MSAColumnAttention()
        self.MSATranslation =  MSATranslation()
        self.OuterProductMean = OuterProductMean()
        self.TriangularMultiplicationIncoming = TriangularMultiplicationIncoming()
        self.TriangularMultiplicationOutgoing = TriangularMultiplicationOutgoing()
        self.TriangularAttentionStartingNode = TriangularAttentionStartingNode()
        self.TriangularAttentionEndingNode = TriangularAttentionEndingNode()
        self.PairTranslation = PairTranslation()


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