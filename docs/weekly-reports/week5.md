# Weekly Report 5

As I spoke with the course instructor this week, he suggested that when the MNIST dataset was created, all digits were at first normalized to a size of 20x20 pixels, and not 28x28.
Doing this prevented the digits from flowing out of the delimited square when they were shifted to put the digits' center of mass to the center of the 28x28 square.
This was a useful tip, since so far I had only paid attention to centering the digits, without touching their dimensions.
To fix this, I started by adding logic to compute the bounding box of each digit. This delimits the area in which each digit is drawn.
Basically, the idea is to draw the smallest possible square around the digit.
I would then clip the digit, and resize it to 20x20 pixels.
And after that, I would center the digit and proceed on as previously. Normalizing the dimensions made a noticeable difference.
The digit 4 for example, which used to be very hard for the model to recognize, became much easier to handle.

Estimated workload:

## Challenges

## What I Learned

Last week I had realized that centering the digits is a crucial step in pre-processing. And this week taught me the importance of standardizing dimensions. Even if the digits are nicely centered on the canvas, we might end up with wildly different dimensions. Someone will draw a huge digit, while someone else will sketch a small one. Hence, normalizing sizes nicely improved prediction accuracy.

## Next Steps
