from flask import Flask, jsonify
from flask_cors import CORS


DEBUG = True
app = Flask(__name__)
CORS(app)

@app.route('/ping', methods=['GET'])
def pong():
    return jsonify('pong!')

@app.route('/prediction/<username>', methods=['GET'])
def make_prediction(username):
    yes = 'making prediction for ' + username
    return jsonify(yes)
    

if __name__ == '__main__':
    app.run(debug=DEBUG)
