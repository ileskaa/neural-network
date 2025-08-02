# User Guide

The fastest way to see this neural network in action is to open
[this link](https://neural-network-jd02.onrender.com/).
It might take some time to load when you open the page for the first.
The cause is that the app is deployed on a free plan, which will go to an idle state if not used for a while.
Once the page is loaded, draw a digit (0-9) in the designated area.
Once you press "Submit", the model will try to recognize the digit and
output its guess to the screen.
The pipeline also involves some pre-processing, and below the prediction,
the app will show the pre-processed canvas.
That is, what the pixels look like before being fed to the network.
You might wonder why the image gets so small after pre-processing.
The reason is that the network was trained on images that are 28x28 pixels in size.
Hence, the digits drawn by the user must be reduced to that same size to be interpretable by the model.
And indeed, on modern screens, 28x28 looks very small.

Do not expect the model to guess every digit correctly; it is definitely not perfect.
But hopefully it'll guess most digits most of the time!

## Training a Model

The deployed model is running on decent parameters, but you can of course train your own model locally.

This project uses Poetry to manage dependencies. If you don't have it yet,
you should start by [installing it](https://python-poetry.org/docs/).

Once poetry is installed, open the project root and run

```bash
poetry install
```

to install dependencies.

Then, still from the project root, use

```bash
poetry run trainmodel
```

to run the model's main file. This will train a new model.

Once the training is finished, a test dataset on 10000 digits will be given to the network.
The accuracy of the model on test data will then be printed to the console.

There are various parameters impacting the training process that can be easily tweaked.
To do so, open `src/model/main.py`.
In there, you can tweak the layers, the number of epochs (training cycles), the learning_rate
and the batch_size.
As a note of caution, increasing the learning_rate too much will trigger overflow errors
during matrix multiplications, and that will cause the model to stop learning.
But feel free to experiment!

To save the parameters obtained during training, you can uncomment the line containing

```python
model.save_parameters()
```

towards the end of the main file.
Saving the parameters will erase the previous ones and those new parameters
will then be used if you decide to run the web app locally.

## Running the App Locally

At the project root, use

```bash
poetry run flask --app src/web/app run
```

to start a local development server on port 5000.

You can then visit <http://localhost:5000/> to view the app.
