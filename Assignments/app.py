import pickle
import numpy as np
import requests
from flask import Flask, request, jsonify

# import libraries
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pickle 

def predict_test(df, dv, model):
    X = dv.transform([df])
    y_pred = model.predict_proba(X)[0, 1]
    return y_pred

with open('card-model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)
    
app = Flask(__name__)

@app.route('/', methods=['POST'])

def predict():
    df = request.get_json()
    prediction = predict_test(df, dv, model)
    card = prediction >= 0.5
    
    result = {
        'card_probability': float(prediction),
        'card': bool(card),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)








