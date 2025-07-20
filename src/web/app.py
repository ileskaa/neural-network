"""A simple flask app to test the model in the browser"""

from flask import Flask, render_template
from model.mlp import MultiLayerPerceptron # pylint: disable=import-error

app = Flask(__name__)

@app.route("/")
def index():
    """The sole page of the Flask app"""
    return render_template('index.html')
