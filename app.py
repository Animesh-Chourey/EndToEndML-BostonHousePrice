import numpy as np
import pandas as pd
import sklearn

from flask import Flask, request, app, jsonify, url_for, render_template

import pickle

# Initialise the flask app
app = Flask(__name__)

# Load the pickle file of the trained LR model
linear_regression_model = pickle.load(open('linear_regression_model.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl','rb'))

# Define the hoempage
@app.route('/')
def home():
    return render_template('home.html')

# Create an API for our predict model
@app.route('/predict_api', methods= ['POST'])
def predict_api():
    data = request.json['data']
    # Get the dict values of key data from json format and get it in a list format
    # Convert it into a numpy array and reshape it into the format needed by the model(here as a 2-d array)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    # Feed the data to the model for prediction
    model_result = linear_regression_model.predict(new_data)
    print(model_result[0])
    return jsonify(model_result[0])

if __name__ == "__main__":
    print(sklearn.__version__)
    app.run(debug = True)