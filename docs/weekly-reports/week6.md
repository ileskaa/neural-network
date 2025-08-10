# Weekly Report 6

This week I was able to get way better results by tweaking the decay rates of my Adam algorithm.
The [paper](https://arxiv.org/pdf/1412.6980) I used as reference recommended a default decay rate of 0.999 for the second moment estimate.
But it was not working for my use case, since my arrays were growing so large, than I would get infinite values, which would stop my model with learning.
I started experimenting with different decay rates, and was able to get really fast convergence by setting the exponential decay rate of my second moment estimate to 0.98.
Doing so, I was able to hit over 98% accuracy on test data in just 5 epochs.
Wayyy faster than with SGD.

Estimated workload:

## Challenges

## What I Learned

## Next Steps
