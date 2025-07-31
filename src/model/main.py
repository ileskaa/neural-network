"""File used to initialize and train the network from scratch"""

from .mlp import MultiLayerPerceptron
from .mnist_loader import load_data
from .nn_utils import normalize_image_data


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = normalize_image_data(x_train)
    x_test = normalize_image_data(x_test)

    epochs = 20

    # On one attempt, this yielded a 97.49% accuracy
    # layers = [784, 256, 128, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.02, batch_size=64)

    # On one attempt, this yielded a 97.44% accuracy
    # layers = [784, 256, 128, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.025, batch_size=92)

    # Was able to hit 97.81% accuracy on normalized data, in 20 epochs
    layers = [784, 384, 128, 10]
    model = MultiLayerPerceptron(layers)
    model.train(x_train, y_train, epochs=epochs, learning_rate=0.02, batch_size=64)

    # Got 97.52% accuracy on normalized data. But training was slooow
    # layers = [784, 512, 64, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.02, batch_size=64)

    # Achieved 97.51% accuracy, but the training time was long
    # layers = [784, 384, 128, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.015, batch_size=48)

    # Hit 97.60% accuracy on mormalized test data
    # layers = [784, 256, 256, 10]
    # model = MultiLayerPerceptron(layers)
    # model.train(x_train, y_train, epochs=20, learning_rate=0.02, batch_size=64)

    model.measure_accuracy(x_test, y_test)

    # Save weights and biases into .npz files
    # model.save_parameters()


if __name__ == "__main__":
    main()
