import numpy as np

def logistic_neurons(W, X):
    z = np.dot(W.T, X)
    g = 1. / (1 + np.exp(-z))
    return g

def linear_neurons(W, X):
    z = np.dot(W.T, X)
    return z

def predictions(W1, W2, X):
    g = logistic_neurons(W1, X)
    z2 = linear_neurons(W2, g)
    return z2

#main
W1 = np.array([[0.2, -0.4, 0.5], [0.3, 0.5, 1]])
W2 = np.array([[2, -1, 5]]).T
X = np.array([[1, 3]]).T
print("W1.shape: " + str(W1.shape))
print("W2.shape: " + str(W2.shape))
print("X.shape: " + str(X.shape))

z = predictions(W1, W2, X)
print("z: " + str(z))