# Weekly Report 6

This week I was able to get way better results by tweaking the decay rates of my Adam algorithm.
The [paper](https://arxiv.org/pdf/1412.6980) I used as reference recommended a default decay rate of 0.999 for the second moment estimate.
But it was not working for my use case, since my arrays were growing so large, than I would get infinite values, which would stop my model with learning.
I started experimenting with different decay rates, and was able to get really fast convergence by setting the exponential decay rate of my second moment estimate to 0.98.
Doing so, I was able to hit over 98% accuracy on test data in just 5 epochs.
Wayyy faster than with SGD.

After that I worked mostly on documentation. My documents should now be up to date.
I also added some types to my functions and methods to improve code readability.

Estimated workload: 12 hours

## What I Learned

I got to understand Adam better. More precisely, I learned about exponential decay rates, which determine how much past gradients influence the current update.
They act as a kind of memory during the training process.
A high value means that the current update will very much depend on what the previous gradients looked like.
The issue I had was that the decay rate for the second moment estimates was too high, which meant that it was not reacting to new gradients fast enough.
This allowed the second moment estimates to get bigger and bigger during the training process, until they achieved such high values that numpy was not able to handle them anymore.
One issue with a big decay rate like that is that if there is a big gradient spike early on, the model will remember it for a very long time, even if subsequent updates are more modest.
This can cause things to go haywire.
Once I lowered that decay rate, the algorithm became more adaptable, which fixed the overflow issue.

## Next Steps

Peer review number 2.
