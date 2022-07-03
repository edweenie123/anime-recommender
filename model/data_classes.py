import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AnimeRatings():
    def __init__(self, ratings_dir='./data/processed/ratings.csv'):
        
        # col_names = ['user_id', 'movie_id', 'score', 'timestamp']
        self.df = pd.read_csv(ratings_dir) 
        
        self.n_users = len(self.df['user_id'].unique())
        self.n_anime = len(self.df['anime_id'].unique())
        
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self, idx):
#         return self.df.iloc[idx]
    
    def split_data(self, test_size=0.1):
        '''Split all the ratings to create a training set and test set'''
        train_set, test_set = train_test_split(self.df, stratify=self.df['user_id'], test_size=test_size)
        return train_set, test_set
    
    def get_all_ratings_from_user(self, user_id):
        pass
        
class AnimeRatingMatrix(Dataset):
    def __init__(self, rating_df, n_users, n_anime, device='cpu'):
        self.n_users = n_users
        self.n_anime = n_anime
        self.matrix = self.generate_interaction_matrix(rating_df, device=device)
        # self.matrix = self.matrix.to(device)
   
    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, idx):
        return self.matrix[idx]
    
    def generate_interaction_matrix(self, rating_df):
        '''Return an user-item interaction matrix using the ratings data frame'''

        matrix = torch.zeros((self.n_users, self.n_anime), dtype=torch.float32)
        
        for row in rating_df.itertuples():
            usr_idx, anime_idx, score = row.user_id, row.anime_id, row.score
            matrix[usr_idx, anime_idx] = score 
        
        return matrix
    
    def load_data(self, batch_size=10):
        return DataLoader(self, batch_size=batch_size, shuffle=True)

class MALAnime():
    def __init__(self, animes_dir='./data/processed/animes.csv'):
        
        self.df = pd.read_csv(animes_dir)
        self.n_anime = len(self.df[''])
    

    def generate_idx_id_dicts(self): 
        '''Generate dictionaries which convert from anime idx to anime id and vice versa'''
        pass


    def ids_to_idxs(self, ids):
        '''Convert all anime ids in <ids> to anime indices'''

        pass

    def idxs_to_ids(self, idxs):
        '''Convert all anime idx in <idx> to anime ids'''
        
        pass

