import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_len, embeddings=500, dropout_rate=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_len, embeddings),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Linear(embeddings, input_len)
        

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def masked_mse(self, pred, tar):
        '''Return the "masked mean squared error" between <pred> and <tar>'''

        mask = (tar > 0).to(torch.float32)

        masked_se = torch.square(mask * (pred - tar))
        masked_mse = torch.sum(masked_se) / torch.maximum(torch.sum(mask), torch.tensor(1))

        return masked_mse
