# Implementation Document

## Project Structure

This project has essentially two parts: the neural network on one side, and a web app on the other. Both are located within the `src/` directory.

### Neural Network

This is definitely the most important part of the project, since it contains the actual implementation of the model. It can be found in `src/model/`. Here's a breakdown of the files:
- `activations.py`
- `loss.py`
- `main.py`
- `mlp.py`: the primary file of the neural network.
- `mnist_loader.py`
- `nn_utils.py`

Then within `src/model/`, there is also the `tests/` directory which contains all unit tests.

### Web App

The web app is a way to showcase the trained network in action. It is located in the `src/web/` directory. Here's a breakdown of its content:
- `parameters/`
- `static/`
- `templates/`
- `app.py`: the Flask server.

## Time and Space Complexities

## Large Language Models (LLMs)

LLMs were a useful tool to quickly get a summary over a specific topic. For example, at the beginning of this project, I hesitated between implementing a multilayer perceptron (MLP) and a convolutional neural network (CNN). I therefore asked ChatGPT to summarize the main differences between these architectures. The answer was quite intelligible, and after watching a few videos to make sure the model was not hallucinating, I decided to go with the MLP. I got the sense that the MLP would be easier to implement, and that made it feel like a more realistic choice.

LLMs were also handy for checking whether some sentence was correct. Sometimes, a sentence I wrote didn't feel quite right, and I would then ask if the sentence was correct. I was able to avoid several mistakes this way.

## References

I mostly used text-based web documents, but some videos also proved useful.

### Documents

- [Weight initilization (Wikipedia)](https://en.wikipedia.org/wiki/Weight_initialization)
- [Rectifier (neural networks) (Wikipedia)](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))
- [Softmax function (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function)
- [Cross-entropy (Wikipedia)](https://en.wikipedia.org/wiki/Cross-entropy)
- [Cross-Entropy in Python](https://github.com/xbeat/Machine-Learning/blob/main/Cross-Entropy%20in%20Python.md)
- [Basic Neural Network from Scratch in Python (Kaggle)](https://www.kaggle.com/code/soham1024/basic-neural-network-from-scratch-in-python)
- [Quickstart (Flask)](https://flask.palletsprojects.com/en/stable/quickstart/)
- [Deploying to Production (Flask)](https://flask.palletsprojects.com/en/stable/deploying/)
- [Adam: a Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980)

### Videos

- [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)
- [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)
