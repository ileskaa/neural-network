# Weekly Report 4

This week I worked on data pre-processing.
I wanted to center the hand-drawn digits on the canvas before feeding them to the network.
Digits of the MNIST dataset are neatly centered, so it made sense that processing my drawings in the same manner would improve accuracy.
And indeed, the results got better.
For some reason, the digit "4" still seems challenging for the network to recognize.
But it now gets most of the digits right most of the time.

Since my web application seemed to function well enough to allow the user to test my neural network, I set out to deploy my model.
I had used render.com in the past for some small hobby projects, and what makes it great is
the ease of deployment and the fact that they offer a free plan.
When deploying a Flask app, one has to choose a Web Server Gateway Interface (WSGI),
which is basically an interface defining how a web server communicates with a Python web app.
I picked Gunicorn, which is the first WSGI listed on Flask's website.
I had some small issues during deployment, like the default Gunicorn port not matching the port in Render's deployment environment, but it all got sorted out pretty quick.
The link to the deployed app can be found in the [user guide](../user-guide.md).

I also began working on the implementation documentation, since I thought it would
be quite painful to write the whole thing at in one go the end of the project.
I might also be hard to remember all the references that have been used at this point.

Estimated workload:

## Challenges

## What I Learned

## Next Steps
