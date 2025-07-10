# Project Specification

This project is implemented as part of the bachelor's degree in computer science of the University of Helsinki. All documentation in this project will be written in English.

The goal is to build a multilayer perceptron (MLP), which is a simple type of neural network. This model will be trained to recognize handwritten digits using the MNIST (Modified National Institute of Standards and Technology) dataset. The program will receive as input a digit written by hand on a 28x28 pixel surface and should be able to recognize that digit. The goal is to achieve at least 97% accuracy.

The aim is the use only Numpy, and possibly Matplotlib, as external libraries. Numpy because it performs vector and matrix operations much faster than if I were to implement them myself. And Matplotlib if I want to plot some charts to illustrate the training process.

At the end of this course, I would like to save my optimized parameters, using [numpy.savez](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) for example, and deploy my model via some cheap cloud provider so that my project can be easily tested. The aim is to create a simple browser-based interface where users can draw digits with mouse or finger, which the model would then recognize.

## Programming Languages

This project is written in Python. The code was tested using Python 3.13.2, but any version >=3.10 should work without issue.

I am also comfortable with Javascript/Typescript, which I use in my daily work. I am therefore able to do peer reviews on projects written in those languages.

## Time and Space Complexity

The layers of the network will be fully connected, i.e., each input neuron will be connected to each output neuron.
Time and space complexity will depend on the number of layers as well as the size of the layers that end up being implemented.
Some estimations of these complexities follow.

### Parameters

Storing the model parameters will require a certain amount of space. For each layer $l$, let $n_l$ be the size (i.e., the number of neurons) of that layer.
For each layer, we have a $n_l \times n_{l-1}$ weight matrix, which thus requires $O(n_l \cdot n_{l-1})$ space.

Let's denote the input layer $l_0$ and let $L$ be the number of layers. Now, the total weight space is
$\sum^L_{l=1} n_l \cdot n_{l-1}$.

The biases require less space since they are vectors instead of matrices. For layer $l$, the bias vector $b_l$ has $n_l$ elements. That makes a total bias space of $\sum_{l=1}^L n_l$.

The weight space is clearly the most demanding of the two, and the space complexity can therefore be simplified to  
$\sum^L_{l=1} n_l \cdot n_{l-1}$.

### Forward Pass

Since each layer is fully connected, a layer with $n$ input neurons and $m$ output neurons will have a time complexity of $O(n\cdot m)$ due to the matrix multiplication involved. To give a concrete example, if the first hidden layer has 784 input neurons and 512 output neurons, that would generate $784 \cdot 512 + 512 = 401920$ operations for a single forward pass. The $+512$ is due to the addition of biases to the ouput neurons. We can thus see that the application of biases has little impact on the time complexity of the layer.

Typically, each subsequent layer will be less computationally heavy due to a narrowing down of layer sizes. Therefore, if the next layer has 512 input and, for example, 256 output neurons, it will perform $512 \cdot 256 +256 = 131328$ operations per sample.

### Backpropagation

Backpropagation is a bit more complex, since it computes gradients of the loss function with respect to weights, inputs and biases by making use of the chain rule.

The gradient w.r.t. weights is computed as $\nabla W = \delta \cdot x^T$ where $\delta$ is the $m$-dimensional error gradient from the next layer (layer $l+1$, if you prefer) and $x^T$ is the transposed input of size $n$. The time complexity of this matrix multiplication would therefore be $O(n\cdot m)$.

The gradient w.r.t. inputs is $\nabla x = W^T \cdot \delta$, where $W^T$ is an $n \times m$ matrix and $\delta$, as mentionned earlier, is $m$-dimensional. Hence, this matrix multiplication would also be $O(n\cdot m)$.

Computing the gradient w.r.t. biases is faster since $\nabla b = \delta$, which is simply $O(m)$.

For the whole layer this yields $O(n\cdot m) + O(n\cdot m) + O(m)$, which simplifies to $O(n\cdot m)$.

### Summing Up

The size of the input will not change, due to the specifications of the MNIST dataset. We hence know that the input layer will have a size of 784. The time complexity will therefore be most impacted by the size of the first hidden layer, which is typically the largest layer after the input layer. This first hidden layer is where the heaviest matrix multiplications will be performed. If performance becomes an issue during training, we can therefore deduce that reducing the size of the first hidden layer from 512 to, say, 256 would lead to a significant speedup.

## Sources I Intend to Use:

- [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio and Aaron Courville
  - especially [Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)

- [Writing automated tests for neural networks](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) by Sebastian Björkqvist

- Videos by Grant Sanderson
  ([3Blue1Brown](https://www.youtube.com/@3blue1brown)):
  - [But what is a neural network? | Deep learning chapter 1](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  - [Gradient descent, how neural networks learn | Deep Learning Chapter 2](https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2)
  - [Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=3)

- Various Wikipedia articles:
  - [Weight initilization](https://en.wikipedia.org/wiki/Weight_initialization)
  - [Rectifier (neural networks)](<https://en.wikipedia.org/wiki/Rectifier_(neural_networks)>)
  - [Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
  - [Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy)
