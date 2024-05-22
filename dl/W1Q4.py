import numpy as np

def perceptron(inputs, weights, threshold):
    weighted_sum = np.dot(inputs, weights)
    return 1 if weighted_sum >= threshold else 0

np.random.seed(42) 
data = np.random.randint(0, 100, size=(500, 3))  

labels = np.array([1 if (age > 28 and salary > 50 and family >= 3) else 0 for age, salary, family in data])

split_index = int(0.7 * len(data))
train_data, test_data = data[:split_index], data[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

weights = np.random.rand(3)
threshold = np.mean([np.dot(x, weights) for x, label in zip(train_data, train_labels) if label == 1])

for epoch in range(10):
    for inputs, label in zip(train_data, train_labels):
        prediction = perceptron(inputs, weights, threshold)
        if label == 1 and prediction == 0:
            weights += inputs
        elif label == 0 and prediction == 1:
            weights -= inputs

test_predictions = [perceptron(x, weights, threshold) for x in test_data]
test_accuracy = sum(test_labels == test_predictions) / len(test_labels)

print(f"Weights: {weights}")
print(f"Threshold: {threshold}")
print(f"Test Accuracy: {test_accuracy*100}")
