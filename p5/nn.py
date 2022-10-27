import numpy as np
import scipy.io as sc
import utils
import logistic_reg as lgr

def feed_forward(theta1, theta2, X):
    a1 = np.c_[np.ones(len(X)), X]          #Size(5000 * 401)
    z2 = np.dot(theta1, a1.T)           
    a2 = lgr.sigmoid(z2)
    a2 = np.c_[np.ones(len(a2[0])), a2.T]   #Size(5000 * 26)
    z3 = np.dot(theta2, a2.T)
    a3 = lgr.sigmoid(z3)                    
    a3 = a3.T                 #Size(5000 * 1)
    return a3

def cost(theta1, theta2, X, y, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """
    p = feed_forward(theta1, theta2, X)
    m = len(X)

    cost = np.sum(y * np.log(p) + (1-y) * np.log(1 - p))

    return -cost/m



def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """


    return (J, grad1, grad2)

def main():
    data = sc.loadmat('data/ex3data1.mat', squeeze_me=True)
    y = data['y']
    y_hot = np.zeros([len(y), 10])
    for i in range(len(y)):
        yMat[i][y[i]] = 1
    X = data['X']

    weights = sc.loadmat('data/ex3weights.mat')
    theta1, theta2 = weights['Theta1'], weights['Theta2']
    c = cost(theta1, theta2, X, y_hot, 0.2)
    print(c)

main()   