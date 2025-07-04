## Solving MNIST as per Chapter 4 of Fastbook

Training a neural network mostly from scratch using Pytorch, to solve MNIST Digit Recognition.

Using `loss.backward()` for getting gradients, and some methods in the manual cross entropy loss function implementation, but mostly everything else should be a direct implementation without library usage for Stochastic Gradient Descent.

The dataset was obtained from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/c/digit-recognizer).

You can see the loss and accuracy against the training dataset in [the notebook](./mnist.ipynb).

The `Further Research` section of this chapter of the book asks you to do your own implementation of `Learner`, a general class that fast.ai uses to handle training.  
The chapter explains how to build a neural net that can tell when an image is either a `3` or a `7`, and then asks you to solve MNIST for all digits.  
This is a repository to solve both problems.

## Kaggle Competition

I did a first submission to the Kaggle Competition using this code, it got 0.9452 [here](https://www.kaggle.com/code/dewstend/mnist-digit-recognizer/notebook).

## Blog

I also tried to comment a bit on my experience in my [blog post](https://dewstend.github.io/blog/fastbook-mnist/).