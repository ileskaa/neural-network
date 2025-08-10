"""File used to initialize and train the network from scratch"""

from .mlp import MultiLayerPerceptron
from .mnist_loader import load_data
from .nn_utils import normalize_image_data


def main():
    """Trains a new model.

    When initializing a new model, layers are passed as a list of integers.
    !!Important: the first and last layers should always be 784 and 10, respectively.
    The reason is that the first layer is the input layer, which receives
    a flattened list of pixel values.
    Since the images are 28x28 pixels in size, it makes 784 pixels in total.
    And the last layer, which is the output layer, should always have size 10
    since the network ouputs an array of discrete probabilities.
    There is one probability for each digit, which makes a total of 10 values.

    The layers between the first and last can however be tweaked according to one's desires.
    You can change layer sizes: e.g., changing 384 to 200.
    Or you can increase the amount of layers: you could for example define
    layers = [784, 384, 128, 64, 10]

    But be aware that if you create huge layers, say a layer of 1000 neurons,
    you might get really long training cycles.

    The `epochs` variable corresponds to the number of training cycles.
    With more training cycles you will usually end up with a more accurate model,
    but training will take longer. There is always a trade-off, right?
    """
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = normalize_image_data(x_train)
    x_test = normalize_image_data(x_test)

    # Stochastic gradient descent (SGD)
    # Was able to hit 97.81% accuracy on normalized data in 20 epochs
    # layers = [784, 384, 128, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.02, batch_size=64)

    # Adaptive moment estimation (Adam)
    # Was able to achieve 98.07% in just 5 epochs
    model = MultiLayerPerceptron([784, 384, 128, 10])
    model.adam(x_train, y_train, alpha=0.001, beta1=0.9, beta2=0.97)

    model.measure_accuracy(x_test, y_test)

    # If you wish to save weights and biases into .npz files:
    # model.save_parameters()


if __name__ == "__main__":
    main()
