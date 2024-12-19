import math
import random
from sklearn.datasets import load_iris

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Normalize dataset
def normalize(data):
    min_vals = [min(col) for col in zip(*data)]
    max_vals = [max(col) for col in zip(*data)]
    normalized = [
        [(x - min_vals[i]) / (max_vals[i] - min_vals[i]) for i, x in enumerate(row)]
        for row in data
    ]
    return normalized

# One-hot encode labels
def one_hot_encode(labels):
    unique_labels = sorted(set(labels))
    encoding = {label: [1 if label == u else 0 for u in unique_labels] for label in unique_labels}
    return [encoding[label] for label in labels]

# Initialize weights
def initialize_weights(rows, cols):
    return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

# Forward pass
def forward_pass(inputs, weights, biases):
    layer_output = []
    for j in range(len(weights[0])):
        z = sum(inputs[i] * weights[i][j] for i in range(len(inputs))) + biases[j]
        layer_output.append(sigmoid(z))
    return layer_output

# Backward pass
def backward_pass(output, target, weights, layer_inputs, learning_rate):
    errors = [(target[i] - output[i]) for i in range(len(target))]
    deltas = [errors[i] * sigmoid_derivative(output[i]) for i in range(len(errors))]

    for i in range(len(layer_inputs)):
        for j in range(len(deltas)):
            weights[i][j] += learning_rate * deltas[j] * layer_inputs[i]
    return deltas

# Train neural network
def train_nn(X, y, input_neurons, hidden_neurons, output_neurons, learning_rate, epochs):
    W1 = initialize_weights(input_neurons, hidden_neurons)
    b1 = [random.uniform(-1, 1) for _ in range(hidden_neurons)]
    W2 = initialize_weights(hidden_neurons, output_neurons)
    b2 = [random.uniform(-1, 1) for _ in range(output_neurons)]

    for epoch in range(epochs):
        for i in range(len(X)):
            inputs = X[i]
            target = y[i]

            # Forward pass
            hidden_output = forward_pass(inputs, W1, b1)
            final_output = forward_pass(hidden_output, W2, b2)

            # Backward pass
            output_deltas = backward_pass(final_output, target, W2, hidden_output, learning_rate)
            backward_pass(hidden_output, [sum(W2[k][j] * output_deltas[j] for j in range(output_neurons)) for k in range(hidden_neurons)], W1, inputs, learning_rate)

    return W1, b1, W2, b2

# Predict
def predict(X, W1, b1, W2, b2):
    predictions = []
    for sample in X:
        hidden_output = forward_pass(sample, W1, b1)
        final_output = forward_pass(hidden_output, W2, b2)
        predictions.append(final_output)
    return predictions

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()
y = iris.target.tolist()

# Normalize features and encode labels
X = normalize(X)
y = one_hot_encode(y)

# Neural network parameters
input_neurons = len(X[0])
hidden_neurons = 5
output_neurons = len(y[0])
learning_rate = 0.1
epochs = 500

# Train the model
W1, b1, W2, b2 = train_nn(X, y, input_neurons, hidden_neurons, output_neurons, learning_rate, epochs)

# Test predictions
predictions = predict(X, W1, b1, W2, b2)
for i, pred in enumerate(predictions):
    predicted_class = pred.index(max(pred))
    actual_class = y[i].index(max(y[i]))
    print(f"Sample {i + 1}: Predicted: {predicted_class}, Actual: {actual_class}")
