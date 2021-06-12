from flask import Flask, request, jsonify
# import numpy as np
# import pandas as pd
import pickle
from recommend import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-type'
filename = 'recommend.pkl'
model = pickle.load(open(filename, 'rb'))


@app.route('/recommendation/<cuisine>', methods=['GET'])
# @cross_origin()
def recommendation(cuisine):
    if request.method == 'GET':
        # cuisine = request.form['cuisine']
        result = model(cuisine).to_dict(orient="records")
        
        return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
