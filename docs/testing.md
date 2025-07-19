# Testing Report

## Unit tests

Go to the project root and use `poetry run pytest src` to run unit tests.

Use   
`poetry run bash -c "coverage run --branch -m pytest src && coverage report -m"`   
to get a coverage report in your terminal.

## Performance in Training

5 attempts for each layer structure. Used 15 epochs:

| Layer structure     | Training time                 |
| ------------------- | ----------------------------- |
| [784, 512, 256, 10] | Between 60.7 and 68.3 seconds |
| [784, 512, 128, 10] | Between 46.2 and 64.5 seconds |
| [784, 256, 128, 10] | Between 24.8 and 31.2 seconds |

As I expected, when writing about time complexity in my project specification, training time decreases significantly when reducing the size of the first hidden layer. The reason is that heaviest matrix multiplications happen at the first hidden layer, given that the input layer, which is typically the largest of layers, feeds into it.
