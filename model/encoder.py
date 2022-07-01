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
        

    def forward(self, input):
        f = self.encoder(input)
        f = self.decoder(f)
        return f

#     def __init__(self, input_len):
        
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_len, 20),
#             nn.ReLU(),
#             nn.Linear(20, 10),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(10, 20),
#             nn.Dropout(0.05),
#             nn.ReLU(),
#             nn.Linear(20, input_len)
#         )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
#     def get_loss(self, model_output, batch):
#         # mask = (batch > 0).to(torch.float32)
#         # model_output = model_output*mask
#         model_output[batch == 0] = 0
        
#         return nn.functional.mse_loss(model_output, batch)
    
    def masked_mse(self, pred, tar):
        mask = (tar > 0).to(torch.float32)

        masked_se = torch.square(mask * (pred - tar))
        masked_mse = torch.sum(masked_se) / torch.maximum(torch.sum(mask), torch.tensor(1))

        return masked_mse
