import torch
import torch.nn as nn


"""
Encoder
    Use RNN Unit (different from paper that use unit not saved on pytorch)
"""
class Encoder(nn.Module):
    def __init__(self, num_words, embedding_size, Tx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = num_words, embedding_dim = embedding_size)
        self.BRNN      = nn.RNN(input_size = embedding_size, hidden_size = Tx, bidirectional = True, batch_first = True)
    
    def forward(self, x) -> torch.Tensor: # (x - batch, sequence length)
        embedded_x = self.embedding(x) # (batch, sequence length, embedding_dim)
        h, _       = self.BRNN(embedded_x) # (batch, sequence length, 2 * Tx)
        return h


"""
Context
    Compute amount of attention
    e = energies -> alpha -> context
"""
def concat_c(prev_s:torch.Tensor, h:torch.Tensor):
    prev_s = prev_s.unsqueeze(1)
    prev_s = prev_s.repeat(1, h.size(1), 1)
    concat = torch.cat((prev_s, h), dim = 2)

    return concat

class Context(nn.Module):
    def __init__(self, prev_s, h): # prev_s (m, Ty)
        super(Context,self).__init__()
        self.Linear     = nn.Linear(in_features = concat_c(prev_s, h).view(-1, prev_s.size(1) + h.size(2)).size(1), out_features = 1000) # (-1, Ty + 2 * Tx)
        self.tanh       = nn.Tanh()
        self.Linear_v   = nn.Linear(in_features = 1000, out_features = prev_s.size(1) + h.size(2), bias = False) # Linear1 out = Linear2 in
        self.Softmax    = nn.Softmax(dim = 1)
        
    def forward(self, prev_s, h):
        e:torch.Tensor = self.Linear(concat_c(prev_s, h).view(-1, prev_s.size(1) + h.size(2)))
        e = self.tanh(e)
        e = self.Linear_v(e)
        e = e.view(h.size(0), h.size(1), -1) # (batch, sequence length, Ty + 2Tx)

        alpha = self.Softmax(e)

        context = torch.matmul(alpha.permute(0,2,1), h)

        return context # (batch, Ty + 2Tx, 2Tx)


"""
Decoder
    Use RNN Unit (different from paper that use unit not saved on pytorch)
    preds : the results that have most probability (save pred on this list) and return
    update prev_s and return
"""
def concat_d(prev_s:torch.Tensor, context:torch.Tensor):
    prev_s = prev_s.unsqueeze(2) # (m, Ty, 1)
    prev_s = prev_s.repeat(1, 1, context.size(2))
    concat = torch.cat((prev_s, context), dim = 1)

    return concat

class Decoder(nn.Module):
    def __init__(self, prev_s, context, Ty):
        super(Decoder, self).__init__()
        self.RNNCell = nn.RNNCell(input_size = concat_d(prev_s, context).view(prev_s.size(0), -1).size(1), hidden_size = Ty)
        self.Linear = nn.Linear(in_features = Ty, out_features = prev_s.size(1))
        self.Softmax = nn.Softmax(dim = -1)

    def forward(self, prev_s, context):
        inputs = concat_d(prev_s, context).view(prev_s.size(0), -1)
        state = self.RNNCell(inputs) # (m, 1000)
        prev_s = state

        preds = self.Linear(state)
        preds = self.Softmax(preds)

        result, _ = torch.max(preds, dim = 1)

        return prev_s, result, preds
        




