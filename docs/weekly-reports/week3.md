# Weekly Report 3

This week I worked on creating a browser interface to test the model. I used Flask since I already had some experience with the framework and it is a quick way to set up a web server using Python. I then used HTML canvas to create a surface where the user can draw using his mouse or finger. The 28x28 pixels that compose the images of the MNIST dataset are however way too small of an area to realistically draw anything on it. I therefore created a surface of 280x280 pixels instead. I then added logic to scale the images down to 28x28 once the user is finished drawing.

Server-side, the neural network is initialized with parameters read from .npz files. Those npz files are generated after model training to persist weights and biases. When the server receives pixel values from the client, they are passed to the model's predict() method, which returns a digit. This digit is then returned to the client for display.

Estimated workload: 14 hours

## Challenges

It wasn't difficult to write methods to save and read parameters to and from .npz files, but interestingly enough, writing unit tests for those turned out to be quite complicated. The tests for the parameter-saving method were easier since I had already used unittest.mock.patch for another test, and I was able to use the same method here. But writing tests for the parameter-loading method was another story. My initial plan was to mock the open() method from Python's `builtins` module to simulate opening a parameter containing file. But I kept running into all kinds of issues. The data saving and loading methods were not playing nicely together in my tests. I tried all kinds of shenanigans but my program kept throwing unexpected errors at my face. After many frustrating attempts, I found out about Python's `tempfile` module, and that saved my day. This module allowed me to create temporary files and directories, which were perfect for my unit tests.

Then I ran into major issues when testing the model in my Flask app. Even though I was getting good results when running the model on MNIST's test dataset, the success rate was really really bad when the model would try to predict the digits that I had drawn myself. It turns out that one reason for the catastrophic success rate was that, by default, HTML canvases have a transparent background. This meant that I was giving my model an image with transparent background and black strokes. The MNIST dataset, on the other hand, is made of white digits against a black background. Hence, I replaced the transparent background with a white one, and flipped the pixel values so that the white background would become black and the black strokes would turn white.

I was now feeding my network with white digits drawn on a black background. That helped, but the accuracy was still nothing close to the 97%-98% I am able to achieve on MNIST's test data. One possible explanation is that while MNIST digits are nicely centered, my hand-drawn digits, by contrast, often end up skewed towards one edge of the canvas. It's very hard to draw a perfectly centered digit by hand. If I can programmatically center the digits before giving them to the network, the accuracy might improve.

## What I Learned

First, I learned the basics of the browser's Canvas API, which I hadn't used before. The API is well documented, which helped a lot. Canvas makes it quite easy to manipulate drawings, like resizing them, and to inspect drawings' pixel values. This was very nice for my use case.

I also learned about the importance of data pre-processing. That step seems way more important than I had imagined. It is very different to test a model on a nicely formatted test dataset than to test it against real world data, which tends to be quite messy. Pre-processing is key in reducing these disparities.

## Next Steps

I want to try to center the digits drawn in the browser to see if that results in better accuracy. I will have to use some maths to find the center of each drawing, and then place that at the center of the canvas.
