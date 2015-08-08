# CUBLAS-ML

This is a library for ML (currently just neural nets) written in CUBLAS and CUDA.

The cpp and h files are the libraries, with exception to main.cpp which shows example usage of the library with the Kaggle bikesharing demand prediction data.
The data has already been edited by me to remove things like headers. The original data can be found at: https://www.kaggle.com/c/bike-sharing-demand/data

Currently, the library is capable of simulating logisitc sigmoid activated artificial neural networks and training them using the momentum method and gradient descent (including batch/stochastic/mini-batch).

I am currently trying to "upgrade" the algorithm to learn from simple batch gradient descent to a hessian method which would be much faster.

The code and algorithm is being "prototyped" in the bikeshare-ml repository hessian branch also under my GitHub account

I am just a 16 year old kid so if you feel that my work deserves it, feel free to send donations to my parents PayPal account:

donghongtian@hotmail.com

Any donations would be greatly welcomed and would really help in encouraging me to work on this project and projects like this. :)
