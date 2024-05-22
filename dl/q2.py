import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.reshape(-1, 1))

# Define sigmoid activation function
def sigmoid(x):
    # Clip the input to prevent overflow
    clipped_x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-clipped_x))


# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define neural network architecture
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Initialize weights and biases for hidden layers
        for i in range(len(hidden_layers)):
            if i == 0:
                self.weights.append(np.random.randn(input_size, hidden_layers[i]))
            else:
                self.weights.append(np.random.randn(hidden_layers[i-1], hidden_layers[i]))
            self.biases.append(np.random.randn(hidden_layers[i]))

        # Initialize weights and biases for output layer
        self.weights.append(np.random.randn(hidden_layers[-1], output_size))
        self.biases.append(np.random.randn(output_size))

    def forward(self, X):
        self.layer_outputs = []
        input_data = X

        # Forward pass through hidden layers
        for i in range(len(self.hidden_layers)):
            input_data = sigmoid(np.dot(input_data, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(input_data)

        # Forward pass through output layer
        output = sigmoid(np.dot(input_data, self.weights[-1]) + self.biases[-1])
        self.layer_outputs.append(output)

        return output

    def backward(self, X, y, learning_rate):
        # Calculate error at output layer
        output_error = y - self.layer_outputs[-1]
        delta = output_error * sigmoid_derivative(self.layer_outputs[-1])

        # Backpropagate error through hidden layers
        for i in range(len(self.hidden_layers), 0, -1):
            self.weights[i] += np.dot(self.layer_outputs[i-1].T, delta) * learning_rate
            self.biases[i] += np.sum(delta, axis=0) * learning_rate
            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(self.layer_outputs[i-1])

        # Update weights and biases for input layer
        self.weights[0] += np.dot(X.T, delta) * learning_rate
        self.biases[0] += np.sum(delta, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)

            if epoch % 10 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')

# Define neural network parameters
input_size = X_train.shape[1]
output_size = y_train_encoded.shape[1]

# Experiment with different architectures
hidden_layers_list = [[2], [5], [10]]

# Train and evaluate MLP for each architecture
for hidden_layers in hidden_layers_list:
    print(f'\nTraining MLP with hidden layers: {hidden_layers}')
    neural_net = NeuralNetwork(input_size, hidden_layers, output_size)
    neural_net.train(X_train, y_train_encoded, epochs=100, learning_rate=0.1)

    # Evaluate model
    predictions = neural_net.forward(X_test)
    accuracy = np.mean(np.argmax(predictions, axis=1) == y_test)
    print(f'Accuracy: {accuracy}')
