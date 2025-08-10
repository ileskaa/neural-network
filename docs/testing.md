# Testing Report

In addition to the neural network this project includes a small web application. The web app is however just aimed at offering an interface for the user to see the model in action. It has therefore been left out of the scope of automatic testing.

## Automatic Testing

Test coverage for the model:
![test coverage](./test_coverage.png)

To run automatic tests, go to the project root and use

```bash
poetry run pytest src
```

To get a coverage report in your terminal, run the following from the project root:

```bash
poetry run coverage run --branch -m pytest src
poetry run coverage report -m
```

### Unit tests

Unit tests are implemented in `/src/model/tests/`.
Most files in this directory are dedicated to unit tests.
The only exception is the file named `/src/model/tests/integration_test.py`,
which contains integration tests. More on those in the next section.

### Integration Tests

Integration tests are implemented in `/src/model/tests/integration_test.py`.
The goal with these tests is to go beyond individual methods and functions, and to test the network as a whole.
This [article](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/) by Sebastian Björkqvist was a big help in implementing integration tests.

## Performance in Training

5 attempts for each layer structure. Used 15 epochs:

| Layer structure     | Training time                 |
| ------------------- | ----------------------------- |
| [784, 512, 256, 10] | Between 60.7 and 68.3 seconds |
| [784, 512, 128, 10] | Between 46.2 and 64.5 seconds |
| [784, 256, 128, 10] | Between 24.8 and 31.2 seconds |

As I expected, when writing about time complexity in my project specification,
training time decreases significantly when reducing the size of the first hidden layer.
The reason is that heaviest matrix multiplications happen at the first hidden layer,
given that the input layer, which is typically the largest of layers, feeds into it.

A batch size of 64 seems to yield better results than a batch size of 128.
Was able to hit a 97% accuracy in 20 epochs with a learning rate of 0.014.
