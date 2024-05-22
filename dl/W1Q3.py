import numpy as np

def mp_neuron(inputs, threshold):
    return 1 if sum(inputs) >= threshold else 0
np.random.seed(42)  
data = np.random.randint(0, 100, size=(500, 3))  
labels = np.array([1 if (age > 28 and salary > 50 and family >= 3) else 0 for age, salary, family in data])

split_index = int(0.7 * len(data))
train_data, test_data = data[:split_index], data[split_index:]
train_labels, test_labels = labels[:split_index], labels[split_index:]

threshold = np.mean([sum(x) for x, label in zip(train_data, train_labels) if label == 1])

accuracies = []
for possible_threshold in range(int(threshold - 10), int(threshold + 10)):
    predictions = [mp_neuron(x, possible_threshold) for x in train_data]
    accuracy = sum(train_labels == predictions) / len(train_labels)
    accuracies.append((possible_threshold, accuracy))

best_threshold, best_accuracy = max(accuracies, key=lambda x: x[1])

test_predictions = [mp_neuron(x, best_threshold) for x in test_data]
test_accuracy = sum(test_labels == test_predictions) / len(test_labels)

print(f"Best Threshold: {best_threshold}")
print(f"Training Accuracy: {best_accuracy*100}")
print(f"Test Accuracy: {test_accuracy*100}")
