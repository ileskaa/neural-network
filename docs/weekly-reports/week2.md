# Weekly Report 2

This week's first task was to create class for my neural network and initialize its weights and biases in the __init__ function. I then implemented methods for the forward pass and backpropagation. The forward pass was pretty easy, but backpropagation was more tricky since I had to keep track of the shapes of all the matrices and vectors, or else matrix multiplications would raise errors. Several times I had to do some calculations using plain old pen and paper not to lose track of the process. I also wrote unit tests for those methods, and then moved on to the prediction. The predict() method was quick to implement since it simply takes the pixel values of an image, or of an array of images, and runs them through the network to try to predict what digit it could be.

Before implementing a method to actually train my network, I needed a way to actually load the MNIST dataset. I had obtained data in UByte format from Kaggle, so I created a module that reads the binary data from thos UByte files and returns data in the form of numpy arrays. I was now ready to tackle the training method of the model, but that proved a bit more tricky. More on that in the [next section](#challenges).

Once I managed to implement a training method that actually worked, I started tweaking various network parameters to see what kind of accuracy I would be able to achieve on the MNIST testing dataset, which is comprised of 10 000 digits. I used 20 epochs for my tests, abd the best score I was able to achieve was a 97.63% accuracy. This was achieved with layers of size [784, 384, 128, 10], a learning rate of 0.02, and a batch size of 64. I don't think I will be able to go past 98% with my simple stochastic gradient descent, but I could potentially get better results using the [Adam](https://arxiv.org/pdf/1412.6980) (adaptive moment estimation) optimization algorithm.

**Estimated workload:**


## Challenges

While last week was mostly smooth sailing, I encountered several problems during this week's coding sessions. The first major issue arose when I had finished writing a first version of my model's training method, which is designed to update model parameters in a way to minimize cross-entropy, which my loss function is based on. I thought I knew what I was doing, but I ran my first training cycle, my reaction was "Holy cow, the cross-entropy is actually increasing". Basically, my training method was doing the exact opposite of what it was supposed to do. After doing some digging, I realized that I needed to mormalize the data I feed to my network. With pixel values ranging from 0 to 255, it is quite hard to find a suitable learning rate, and my learning of 0.01 was causing my gradients to overshoot. This means that the gradients were shifting my weights and biases so violently in the opposite direction that they would miss the mark and actually worsen the model. After I scaled all the pixel values down to [0, 1], my method started playing out nicely.

But then arose another issue that made me scratch my head.   
Since my training method seemed to work, I started playing with my model by testing out different learning rates, numbers of layers, and layer sizes. Until I suddenly started getting errors about overflows caused by matrix multiplications. Training would go smoothly for the first few epochs, until my model would throw this error at me, learning would halt, and I would end up with a horrible model unable to predict the slightest digit. I ran the training method several times, but got the same results. I began looking at my backpropagation and training methods, but couldn't find anything wrong with them. It took me some time to realize that if my learning rate is too high, my gradients could grow so big that my computer was not able to perform matrix multiplications with them anymore.
I also noticed that if I go from 2 to 3 hidden layers, this issue would happen much more often, since it would increase the amount of multiplications performed in the network. There is a fine balance to strike between the number of layers and the training rate.


## What I Learned

The issues I encountered while playing around with my model's layers and learning rate showed me how there is a fine balance to strike between the number of layers and the training rate. If there are too many layers or if the learning rate is too high, the system would break and stop learning. On the other hand, if the learning rate is too low, the model will not learn as fast as it could, and might need a large number of epochs to get decent results.

My expectation was that 3 hidden layers would yield better results that 2, at the expense of training duration. The training time did indeed increase, but the 3 layers actually produced slightly worse results. Clearly, more is not always better when it comes to neural nets.

Another surprise, was that even a single hidden layer can produce quite good results if it is big enough. A hidden layer of size 512 for example. Not quite as good as 2 hidden layers, but close.

Another realization was that batch sizes also affect the propensity of matrix multiplications to overflow. A smaller batch size will be more prone to overflow. A smaller batch size therefore needs to be balanced with a smaller learning rate. My hypothesis is that the increased stochasticity caused by lowering the batch size increases the probability of producing large gradients. And these large gradients mean trouble when it comes to matrix multiplications. In effect, when the batch size is reduced, each gradient gets smaller, but the network need to perform more matrix multiplications total. This seems to cause some gradients to go wild.

I also learned how to use unit tests to verify that strings are printed out correctly. The unittest library has a nice method called patch, which can be used to decorate another function. Thanks to patch(), we can redirect the standard output to an object that can then be uses to verify what would have been printed by the method we are testing.


## Next Steps

Pillow library?