# Implementation Document

## Project Structure

This project has essentially two parts: the neural network on one side, and a web app on the other. Both are located within the `src/` directory.

### Neural Network

This is definitely the most important part of the project, since it contains the actual implementation of the model. It can be found in `src/model/`. Here's a breakdown of the files:

- `activations.py`: contains activation functions as well as their derivatives
- `loss.py`: implements the loss function and its gradient
- `main.py`: the module that is run to train a new model
- `mlp.py`: the primary file of the neural network. Contains the `MultilayerPerceptron` class, which has the necessary methods to train and evaluate a model
- `layer.py`: contains the Layer class, which is used to keep track of weights, biases, and moment estimates when running the adaptive moment estimation algorithm
- `mnist_loader.py`: helper module to load the MNIST dataset
- `nn_utils.py`: several utility functions, used for example in weight initialization and to one-hot encode values

Within `src/model/`, there is also the `tests/` directory which contains all automatic tests.

### Web App

The web app is a way to showcase the trained network in action. It is located in the `src/web/` directory. Here's a breakdown of its content:

- `parameters/`: contains `.npz` files, which are used to store the model parameters
- `static/`: contains static files like Javascript and CSS
- `templates/`: has the HTML template of the page
- `app.py`: the Flask server

## Time and Space Complexities

The network is fully connected, meaning that each node of each layer is connected to each node of the previous and next layers. Time and space complexities therefore depend on the number of hidden layers as well as their size. The size of the input and output layers are constant, since each digit the network receives consists of 784 pixel values, and the network will always output a discrete probability distribution of 10 values.

Let $n$ be the number of input neurons to a layer and let $m$ be the number of output neurons. The complexity for each layer will then be $O(n\cdot m + m)$,
which simplifies to $O(n\cdot m)$. The product $n\cdot m$ is due to the matrix multiplications that have to be performed at each layer. The $+m$ is due to the biases that get added at each layer.
The impact of the addition of biases on time and space complexity is however negligible.

## Performance in Training

I collected training durations for different layer structures. The results follow.

5 attempts for each layer structure. Stochastic gradient descent. Used 15 epochs:

| Layer structure     | Training time                 |
| ------------------- | ----------------------------- |
| [784, 512, 256, 10] | Between 60.7 and 68.3 seconds |
| [784, 512, 128, 10] | Between 46.2 and 64.5 seconds |
| [784, 256, 128, 10] | Between 24.8 and 31.2 seconds |

As I suspected while writing about time complexity in my project specification,
training time decreases significantly when reducing the size of the first hidden layer.
The reason is that the heaviest matrix multiplications happen at the first hidden layer,
given that the input layer, which is typically the largest, feeds into it.

## Model accuracy

Adaptive moment estimation (Adam) yielded the best results. Each run will give a bit different
results since weights are initialized randomly and the dataset is shuffled during
training to induce stochasticity. But with Adam, 3 epochs are generally sufficient to
achieve an accuracy above 97% on the test dataset.
Running 9 epochs should get you past the 98% mark.
With some luck, that benchmark can however be reached much sooner.
On one run, I was even able to get 98.07% in just 5 cycles, while using a 0.98 beta-2
decay rate.

With stochastic gradient descent, achieving over 97% accuracy will typically take
around 12 epochs. However, going past the 98% mark proves pretty much impossible.
Even with 30 epochs, I was not able to reach 98%.

## Flaws and Possible Improvements

If I had more time, it would have been nice to implement more training methods.
Currently my project has stochastic gradient descent (SGD) and Adam.
But there are so many more options! Like SGD with momentum and root mean square propagation, just to name a few.

Also a CLI might have been a nice addition. Right now the user can tweak parameters by modifying the arguments given to the training method in the main file. But it could have been nice to have a CLI to guide the user and to allow him to customize parameters in a more interactive way.

## Large Language Models (LLMs)

LLMs were a useful tool to quickly get a summary over a specific topic. For example, at the beginning of this project, I hesitated between implementing a multilayer perceptron (MLP) and a convolutional neural network (CNN). I therefore asked ChatGPT to summarize the main differences between these architectures. The answer was quite intelligible, and after watching a few videos to make sure the model was not hallucinating, I decided to go with the MLP. I got the sense that the MLP would be easier to implement, and that made it feel like a more realistic choice.

LLMs were also handy for checking whether some sentence was correct. Sometimes, a sentence I wrote didn't feel quite right, and I would then ask if the sentence was correct. I was able to avoid several grammatical mistakes this way.

## References

I mostly used text-based web documents, but some videos also proved useful.

### Documents

- [Weight initialization (Wikipedia)](https://en.wikipedia.org/wiki/Weight_initialization)
- [Rectifier (neural networks) (Wikipedia)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax function (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function)
- [Cross-entropy (Wikipedia)](https://en.wikipedia.org/wiki/Cross-entropy)
- [Cross-Entropy in Python](https://github.com/xbeat/Machine-Learning/blob/main/Cross-Entropy%20in%20Python.md)
- [Basic Neural Network from Scratch in Python (Kaggle)](https://www.kaggle.com/code/soham1024/basic-neural-network-from-scratch-in-python)
- [Quickstart (Flask)](https://flask.palletsprojects.com/en/stable/quickstart/)
- [Deploying to Production (Flask)](https://flask.palletsprojects.com/en/stable/deploying/)
- [Adam: a Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)
- [Adam Optimizer Tutorial: Intuition and Implementation in Python](https://www.datacamp.com/tutorial/adam-optimizer-tutorial)
- [The MNIST database of handwritten digits](https://web.archive.org/web/20200430193701/http://yann.lecun.com/exdb/mnist/)
- [Dataset Card for MNIST](https://huggingface.co/datasets/ylecun/mnist)
- [Writing automated tests for neural networks](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)

### Videos

- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)
- [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
