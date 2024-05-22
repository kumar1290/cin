import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def train_perceptron(X, y, learning_rate=0.01, epochs=100):
    weights = np.zeros(X.shape[1])
    bias = 0
    for epoch in range(epochs):
        errors = 0
        for xi, target in zip(X, y):
            update = learning_rate * (target - perceptron(xi, weights, bias))
            weights += update * xi
            bias += update
            errors += int(update != 0.0)
        if errors == 0:
            print(f"Converged after {epoch+1} epochs")
            break
    return weights, bias

def perceptron(inputs, weights, bias):
    return np.where(np.dot(inputs, weights) + bias >= 0, 1, 0)

def plot_decision_boundary(X, y, weights, bias, ax, title):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = perceptron(np.c_[xx.ravel(), yy.ravel()], weights, bias)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-1,0,1], alpha=0.2, colors=['blue','red'])
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

np.random.seed(0)
X = np.array([[0, 0.1], [0, 10], [1, 1], [1, 100]])
y = np.array([1, 0, 0, 1])

print("Training without normalization:")
weights, bias = train_perceptron(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nTraining with normalization:")
weights_scaled, bias_scaled = train_perceptron(X_scaled, y)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plot_decision_boundary(X, y, weights, bias, ax[0], "Without Normalization")
plot_decision_boundary(X_scaled, y, weights_scaled, bias_scaled, ax[1], "With Normalization")
plt.show()
