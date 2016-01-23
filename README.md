# CUBLAS-ML

This is a library for ML (currently just neural nets) written in CUBLAS and CUDA.

I wrote this library to practice my knowledge of ML, C++ & CUDA.

The cpp and h files are the libraries, with exception to main.cpp which shows example usage of the library with the Kaggle bikesharing demand prediction data and another example for classification using the MNIST data.
The data has already been edited by me to remove things like headers. The original data can be found at:
https://www.kaggle.com/c/bike-sharing-demand/data
https://www.kaggle.com/c/digit-recognizer/data

Currently, the library is capable of simulating feed forward multilayer perceptrons and training them using the momentum method and gradient descent (including batch/stochastic/mini-batch).

I have coded activations functions for hyperbolic tangent, sigmoid, and softmax outputs. I have also add coded the negative logarithmic maximum likelihood, the cross entropy and the squared error cost functions. However the code has been re-written to allow easy addition of new ones using function pointers.
