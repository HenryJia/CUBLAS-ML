# CUBLAS-ML

This is a library for ML (currently just neural nets) written in CUBLAS and CUDA.

The cpp and h files are the libraries, with exception to main.cpp which shows example usage of the library with the Kaggle bikesharing demand prediction data and another example for classification using the MNIST data.
The data has already been edited by me to remove things like headers. The original data can be found at:
https://www.kaggle.com/c/bike-sharing-demand/data
https://www.kaggle.com/c/digit-recognizer/data

Currently, the library is capable of simulating logisitc sigmoid activated artificial neural networks and training them using the momentum method and gradient descent (including batch/stochastic/mini-batch).

I am currently trying to also add a hessian training method (Quasi-Newton) which may be much faster.

I am just a 16 year old kid so if you feel that my work deserves it, feel free to send donations to my parents PayPal account:

donghongtian@hotmail.com

Any donations would be greatly welcomed and would really help in encouraging me to work on this project and projects like this. :)
