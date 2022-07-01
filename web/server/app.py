from flask import Flask, jsonify
from flask_cors import CORS
import requests

# This file contains my API KEYS and SECRETS
import config

DEBUG = True
app = Flask(__name__)
CORS(app)

@app.route('/ping', methods=['GET'])
def pong():
    return jsonify('pong!')

@app.route('/')
def test():
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
    anime_list = []

    for anime in res.json()['data']:
        src = anime['node']['main_picture']['medium']
        title = anime['node']['title']
        score = anime['list_status']['score']

        anime_obj = {
                'imgSrc' : src,
                'title' : title,
                'predScore' : score
                }

        anime_list.append(anime_obj)

    return jsonify(anime_list)
    # return yes
    

if __name__ == '__main__':
    app.run(debug=DEBUG)
