from flask import Flask, jsonify
from flask_cors import CORS
import requests
import torch

import sys
sys.path.append('../../model')

from data_classes import MALAnime
from encoder import Encoder

# This file contains my API KEYS and SECRETS
import config

DEBUG = True
app = Flask(__name__)
CORS(app)

mal_anime = MALAnime('../../model/data/processed/animes.csv')
net = Encoder(mal_anime.n_anime, 500)
net.load_state_dict(torch.load('../../model/model_save.pth', map_location=torch.device('cpu')))

@app.route('/ping', methods=['GET'])
def pong():
    return jsonify('pong!')

@app.route('/')
def test():
    print(mal_anime.get_anime_info(0))
    return config.API_KEY

API_URL = 'https://api.myanimelist.net/v2'
REQUEST_HEADERS = { 'Authorization' : 'Bearer ' + config.ACCESS_TOKEN }

@app.route('/prediction/<username>', methods=['GET'])
def make_prediction(username):
    params = { 'fields' : 'list_status',
            'limit' : '100'
            }

    res = requests.get(API_URL + f'/users/{username}/animelist', headers=REQUEST_HEADERS, params=params)
    # print(res.json())

    res_data = res.json()['data']

    # for anime in res_data:
    #     src = anime['node']['main_picture']['medium']
    #     title = anime['node']['title']
    #     score = anime['list_status']['score']
    #     anime_id = anime['node']['id']
    #     link = f'https://myanimelist.net/anime/{anime_id}'

    #     anime_obj = {
    #             'imgSrc' : src,
    #             'title' : title,
    #             'predScore' : score,
    #             'link' : link
    #             }

    #     anime_list.append(anime_obj)

    user_tensor = create_user_tensor(res_data)
    print(user_tensor)
    # pred_ids = mal_anime
    pred_scores, rec_idxs = net.make_recommendations(user_tensor, 25)
    # rec_ids = mal_anime.idxs_to_ids(rec_idxs)
    
    
    anime_recs = []

    for score, idx in zip(pred_scores, rec_idxs):
        # print(idx)
        # break
        anime_info = mal_anime.get_anime_info(idx)
        anime_info['predScore'] = score

        anime_recs.append(anime_info)


    return jsonify(anime_recs)
    # return yes

def create_user_tensor(anime_list_obj):

    user_array = [0 for i in range(mal_anime.n_anime)]
    
    for anime in anime_list_obj:
        anime_id = anime['node']['id']
        anime_idx = mal_anime.convert_id_to_idx(anime_id)
        
        # filter animes that are not in our dataset
        if anime_idx == -1:
            continue

        score = anime['list_status']['score']

        user_array[anime_idx] = score

        
    return torch.FloatTensor(user_array)



if __name__ == '__main__':
    app.run(debug=DEBUG)
