import numpy as np
import matplotlib.pyplot as plt

def train_perceptron(X, y, learning_rate=0.1, epochs=10):
    weights = np.zeros(X.shape[1])
    bias = 0

    for _ in range(epochs):
        for xi, target in zip(X, y):
            update = learning_rate * (target - perceptron(xi, weights, bias))
            weights += update * xi
            bias += update
    return weights, bias

def perceptron(inputs, weights, bias):
    return np.where(np.dot(inputs, weights) + bias > 0, 1, 0)

def plot_decision_boundary(X, y, weights, bias, title):
    plt.scatter(X[:, 0], X[:, 1], c=y)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = perceptron(np.c_[xx1.ravel(), xx2.ravel()], weights, bias)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.2)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

np.random.seed(1)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  

init_weights = np.random.rand(2)
init_bias = np.random.rand(1)

plot_decision_boundary(X, y, init_weights, init_bias, "Before Training")

weights, bias = train_perceptron(X, y)

plot_decision_boundary(X, y, weights, bias, "After Training")

'''
The train_perceptron function initializes the weights and bias to zero and then iteratively adjusts the weights and bias based on the perceptron learning rule across the number of epochs. The perceptron learning rule updates the weights and bias based on the error between predicted and actual outputs with a learning rate that determines the size of the step taken at each iteration.


Before and after training, the decision boundary is plotted by using plot_decision_boundary function, which visualizes the data points and the decision boundary computed by the perceptron's weights and bias.


However, since the XOR problem is not linearly separable, even after training, the perceptron will not be able to find a proper decision boundary that solves the XOR problem. It will likely converge to a point where it performs no better than random guessing. This is precisely why a single-layer perceptron cannot solve the XOR problem and why more complex architectures like multi-layer networks (which can represent non-linear boundaries) are necessary for such problems.

'''