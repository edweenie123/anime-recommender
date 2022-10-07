import torch
import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from data_classes import AnimeRatings, AnimeRatingMatrix, MALAnime
from encoder import Encoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the train and test sets
ratings = AnimeRatings()
train_df, test_df = ratings.split_data(test_size=0.1)

train_mat = AnimeRatingMatrix(train_df, ratings.n_users, ratings.n_anime, device)
test_mat = AnimeRatingMatrix(test_df, ratings.n_users, ratings.n_anime, device)

train_iter = train_mat.load_data(batch_size=256)

# create the model
net = Encoder(ratings.n_anime, 500, dropout_rate=0.73)
net = net.to(device)
trainer = torch.optim.Adam(net.parameters(), lr = 0.0001)
criterion = nn.MSELoss()
epochs = 500


def train():
    net.train() # turn out dropout

    for epoch in range(epochs):
        
        total_error = 0
        cnt = 0
        
        
        for x in train_iter: 
            tar = x
            pred = net(tar)

            loss = net.masked_mse(pred, tar)
           
            with torch.no_grad():
                total_error += torch.sqrt(loss)
                cnt += 1
                
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            
        entire_loss = total_error / cnt

        if (epoch + 1) % 25 == 0:
            print(f'epoch {epoch + 1}, loss {entire_loss}'

def test():
    pass

# export the trained model
torch.save(net.state_dict(), './model_save.pth')

if __name__ == '__main__':
    train()

