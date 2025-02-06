import torch
import torch.nn as nn
import torch.optim as optim

from Model import Encoder, Context, Decoder
from Loss import Loss

Embedding_size = 1000
Batch_size     = 80
learning_rate  = 0.01

class train_model(nn.Module):
    def __init__(self):
        super(train_model, self).__init__()
        
    def forward(self, x, y, batch, Ty): # Tx will be on code automatically after Embedding

        # x should be pre-processed first
        encoder = Encoder(num_words = len(x), embedding_size = Embedding_size) # suppose x is preprocessed data
        context = Context(prev_s, encoded_x)
        decoder = Decoder(prev_s, context_x, Ty)

        optimizer = optim.Adam(params = list(encoder.parameters()) + list(context.parameters()) + list(decoder.parameters()), lr = learning_rate)
        
        encoded_x = encoder.forward(x) # h on paper

        s0 = torch.randn(batch, Ty)
        prev_s = s0

        results = []

        for epoch in range(Ty):

            
            context_x = context.forward(prev_s, encoded_x)

            
            prev_s, result, preds = decoder.forward(self, prev_s, context_x)
            
            # result should be post-processed
            results.append(result) # suppose result is postprocessed data

            loss = Loss()
            total_loss = loss(y, preds)

            optimizer.zero_grad
            total_loss.backward()
            optimizer.step()

        translated = ''.join(results)
        print(translated)

if __name__ == "__main__":
   train_model()



