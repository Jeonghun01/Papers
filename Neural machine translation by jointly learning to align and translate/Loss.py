import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, y, preds):
        loss = F.mse_loss(y, preds)
        return loss