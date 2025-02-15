import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from Model import Encoder, Decoder

token_size    = 25000
d_model       = 512
step_num      = 100
warmup_step   = 4000
learning_rate = (d_model ** -0.5) * min(step_num**-0.5, step_num * (warmup_step ** -1.5))
beta1         = 0.9
beta2         = 0.98
epsilon       = 1e-9


class train(nn.Module):
    def __init__(self,):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, y):
        # suppose x, y are preprocessed dataset
        x = DataLoader(x, batch_size = token_size, shuffle = False)

        start_token = '<SOS>'

        optimizer = opt.Adam(lr = learning_rate, beta1 = beta1, beta2 = beta2, eps = epsilon)

        for epoch in range(step_num):
            encoder_result = self.encoder(x)

            pre_output_softmax = self.decoder(start_token, encoder_result)
            loss_softmax = pre_output_softmax

            pre_output_token = torch.argmax(pre_output_softmax, dim=-1)
            pre_output_token = pre_output_token.max(dim=1)[0].unsqueeze(1)

            tokens = pre_output_token

            while(1):
                output_softmax = self.decoder(pre_output_token, encoder_result)
                loss_softmax = torch.cat((pre_output_softmax, output_softmax), dim = 0)

                output_token = torch.argmax(output_softmax, dim=-1)
                output_token = output_token.max(dim=1)[0].unsqueeze(1)
                tokens = torch.cat((pre_output_token, output_token), dim = 1)

                count = 0
                for i in token_size:
                    for r in len(tokens.size(1)):
                        if '<EOS>' in tokens[i]:
                            count += 1
                            if count == token_size:
                                break

                pre_output_softmax = output_softmax
                pre_output_token = output_token

            loss = nn.CrossEntropyLoss(y, loss_softmax)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                

if __name__ == "__main__":
   train()



