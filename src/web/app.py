"""A simple flask app to test the model in the browser"""

import numpy as np
from flask import Flask, render_template, request
from model.mlp import MultiLayerPerceptron  # pylint: disable=import-error
from model.nn_utils import load_parameters

app = Flask(__name__)

weights, biases = load_parameters()
model = MultiLayerPerceptron(weights=weights, biases=biases)


@app.route("/")
def index():
    """The sole page of the Flask app"""
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def post():
    """Receives pixel values from the canvas and returns a prediction"""
    data = request.get_json()
    print('pixels', data['digit'])
    pixel_values = data['digit']
    normalized = np.array(pixel_values) / 255
    pred = model.predict(normalized)
    print('pred', pred)
    return str(pred)
