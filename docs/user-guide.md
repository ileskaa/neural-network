# User Guide

The fastest way to see this neural network in action is to open
[this link](https://neural-network-jd02.onrender.com/).
It might take some time to load when you open the page for the first time.
The reason is that the app is deployed on a free plan, which will go to an idle state
if not used for a while.
Once the page is loaded, draw a digit (0-9) in the designated area.
Once you press "Submit", the model will try to recognize the digit and
output its guess to the screen.
The pipeline also involves some pre-processing, and below the prediction,
the app will show the pre-processed canvas.
That's what the pixels look like just before being fed to the network.
You might wonder why the image gets so small after pre-processing.
The reason is that the network was trained on images that are 28x28 pixels in size.
Hence, the digits drawn by the user must be reduced to that same size to be
interpretable by the model.
And indeed, on modern screens, 28x28 looks very small.

Do not expect the model to classify every single input correctly;
it is definitely not perfect.
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

Once the training is finished, a test dataset of 10000 digits will be given to the network.
The accuracy of the model on test data will then be printed to the terminal.

There are various parameters impacting the training process that can be easily tweaked.
To do so, open `src/model/main.py`.

### Tweaking the layers

In `src/model/main.py` you can tweak the layers of the model.
But note that first and last layers should not be changed, as that would result in an error
during training. The input layer should always have size 784 since that's the number of
pixels in an image of the MNIST dataset. And the ouput layer should always have size 10 since
there are 10 possible digits.
The layer in between are called hidden layers and can take any non-zero positive value.
But be aware that if you create very large layers, training will take forever.
The example of the main file uses only 2 hidden layers, but you can create
as much of them as you want. There must however be at least be one hidden layer,
since otherwise the model won't learn anything.

### Adam

By default the network will be trained using adaptive moment estimation (Adam) since that
is the most effective training method.

Adam accepts the following parameters:
- x: an array of images represented by their pixel values
- y: array of digits corresponding to the images
- loss_goal: once the loss gets below this value, we consider the model to have converged
- epochs: the number of training cycles. However, if the model achieves its loss goal
  before the given number of epochs, training will end
- alpha: the learning rate
- beta1: exponential decay rate for first moment estimates. The first moment is the mean
  of the gradient
- beta2: exponential decay rate for second raw moment estimates. The second raw moment is the
  uncentered variance of the gradient, i.e. the mean of the squared gradient components
- batch_size: training data is split into batches to induce stochasticity. This controls
  how big each of those batches are

The higher the decay rates, the more the model will "remember" the previous gradient updates.
Adam updates parameters based on moving averages, and those decay rates determine how
much of an impact each new gradient will have.
This makes the training smoother as it avoids big spikes.
An exponential decay rate of 0.9,
for example, means that the update is 90% based on previous updates, and 10%
comes from the recently computed gradient.

### Stochastic Gradient Descent (SGD)

Besides Adam, you can train a model using stochastic gradient descent.
SGD also achieves good results, but training takes slightly longer than with Adam.
More epochs are required when using Adam, but on the other hand, each epoch is shorter.
The main file includes a commented out line that allows you to run SGD.
You can simply comment the line that includes the `model.adam` method, and uncomment the line
having the `model.train` method.

In case you want to tweak the knobs of the `train` method,
it accepts the following parameters:
- x_train: image pixel values
- y_train: actual digits corresponding to pixel values
- epochs: number of training cycles
- batch_size: SGD splits training data into batches and this allows you to control
  how big each batch is
- learning_rate: how much each gradient will impact parameter updates

### Saving parameters

To save the parameters obtained during training, you can uncomment the line containing

```python
model.save_parameters()
```

towards the end of the main file.

Saving parameters will erase the previous ones. As a result those new parameters
will then be used if you decide to run the web app locally.

The parameters are saved into `.npz` files, which is binary format implemented by numpy.

## Running the Web App Locally

From the project root, use

```bash
poetry run flask --app src/web/app run
```

to start a local development server on port 5000.

You can then visit <http://localhost:5000/> to view the app.
