import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, num_words, embedding_size, Tx):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = num_words, embedding_dim = embedding_size)
        self.BRNN = nn.RNN(input_size = embedding_size, hidden_size = Tx, bidirectional = True, batch_first = True)
    
    def forward(self, x) -> torch.Tensor: # (x - batch, sequence length)
        embedded_x = self.embedding(x) # (batch, sequence length, embedding_dim)
        h, _ = self.BRNN(embedded_x) # (batch, sequence length, 2 * Tx)
        return h

def concat(prev_s:torch.Tensor, h:torch.Tensor):
    prev_s = prev_s.unsqueeze(1)
    prev_s = prev_s.repeat(1, h.size(1), 1)
    concat = torch.cat((prev_s, h), dim = 2)

    return concat


class Context(nn.Module):
    def __init__(self, prev_s, h):
        super(Context,self).__init__()
        self.Linear = nn.Linear(in_features = concat(prev_s, h).view(-1, prev_s.size(1) + h.size(2)).size(1), out_features = 1000) # (-1, Ty + 2 * Tx)
        self.tanh = nn.Tanh()
        self.Linear_v = nn.Linear(in_features = 1000, out_features = prev_s.size(1) + h.size(2), bias = False) # Linear1 out = Linear2 in
        
    def forward(self, prev_s, h):
        e:torch.Tensor = self.Linear(concat(prev_s, h).view(-1, prev_s.size(1) + h.size(2)))
        e = self.tanh(e)
        e = self.Linear_v(e)
        e = e.view(h.size(0), h.size(1), -1)

        # todo... alpha -> context

class Decoder(nn.Module):
    pass

