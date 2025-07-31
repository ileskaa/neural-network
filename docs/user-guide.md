# User Guide

## Web App

The fastest way to see this neural network in action is to open
[this link](https://neural-network-jd02.onrender.com/)
and draw a digit in the designated area.
Once you press "Submit", the model will try to recognize the digit and
output its guess to the screen.
The pipeline also involves some pre-processing, and below the prediction,
the app will show the pre-processed canvas.
That is, what the pixels look like before being fed to the network.

## Training a Model

From the project root, use `poetry run trainmodel` to run the model's main file.
This will train a new model.

## Running the App Locally

Use

```bash
poetry run flask --app src/web/app run
```

to start a local development server on port 5000.

You can then visit <http://localhost:5000/> to view the app.
