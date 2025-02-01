import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self, ):
        super(ContentLoss, self).__init__()
        pass

    def forward(self, C:torch.Tensor, G:torch.Tensor):
        # do not need reshape necessary
        loss = F.mse_loss(C, G)
        return loss
        

class StyleLoss(nn.Module):
    def __init__(self, ):
        super(StyleLoss, self).__init__()
        pass

    def gram_matrix(self, x:torch.Tensor):
        """
            x shape - (b, c, H, W)
            reshape -> (b, c, H*W) = (b, N, M)
            transpose -> (b, M, N)
            S - (b, N, N)
        """
        b, c, H, W = x.size()
        x = x.view(b, c, H * W)
        x_t = x.transpose(dim0 = 1, dim1 = 2)
        S = torch.matmul(x, x_t)
        return S.div(4 * b * c * H * W) # normalization

    def forward(self, S, G):
        S = self.gram_matrix(S)
        G = self.gram_matrix(G)
        loss = F.mse_loss(S, G)
        return loss
        