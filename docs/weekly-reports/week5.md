# Weekly Report 5

As I spoke with the course instructor this week, he suggested that when the MNIST dataset was created, all digits were at first normalized to a size of 20x20 pixels, and not 28x28.
Doing this prevented the digits from flowing out of the delimited square when they were shifted to put the digits' center of mass to the center of the 28x28 square.
This was a useful tip, since so far I had only paid attention to centering the digits, without paying attention to their dimensions.
To fix this, I started by adding logic to compute the bounding box of each digit. This box delimits the area in which each digit is drawn.
Basically, the idea is to draw the smallest possible square around the digit.
I then clip the digit, and resize it to 20x20 pixels.
After that, I center the digit and then proceed on as usual. Normalizing the dimensions made a noticeable difference.
The digit 4 for example, which used to be very hard for the model to recognize, became much easier to handle.

After working on pre-processing, I proceeded to do the peer review.
The student had done some really neat work and had built a nice CLI for his app.

I also started working on integration tests. During previous weeks I had only focused on unit tests,
but now I also have tests to verify that the different components of the system play well together.

Finally, I continued working on the Adam optimization algorithm. I managed to train a model with it, but so far the results haven't been very impressive. I am able to reach a good accuracy, but the training actually takes longer than with stochastic gradient descent.

Estimated workload: 15 hours

## Challenges

The Adam algorithm definitely posed challenges. It took me quite a while to implement it. I had to deal with a bunch of errors, since some vector values where growing to infinity during training. I was able to solve that by implementing a clipping mechanism to bound the values to a reasonable range.

## What I Learned

Last week I had realized that centering the digits is a crucial step in pre-processing. And this week taught me the importance of standardizing dimensions. Even if the digits are nicely centered on the canvas, we might end up with wildly different dimensions. Someone will draw a huge digit, while someone else will sketch a small one. Hence, normalizing sizes nicely improved prediction accuracy.

I actually learned a lot doing the peer review. More than I expected.
The project I reviewed had really nicely documented functions.
This gave me ideas on how to improve the docstrings in my own project.
Also the use of types in the project really made the code more understandablem which made me realize that I should be using them more myself.

I also got to know more about Adam, and how the first and second moment estimates must be maintained for each layer separately.
It is important to watch how the moment estimate vectors and matrices evolve during the training process, since one can quite easily reach such high values that they are interpreted as infinity by numpy, which is obviously not good for the training process.

## Next Steps

I want to experiment with the parameters of the Adam algorithm to see if I can make the network converge faster.
