import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0]).reshape(-1, 1) # The .reshape(-1, 1) reshapes y into a column vector so that it matches the format expected by the neural network.

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Neural Network Training Function
def train_xor_nn(X, y, learning_rate=0.1, epochs=10000):
    np.random.seed(42)  # For reproducibility

    # Initialize weights and biases
    input_neurons = X.shape[1]
    hidden_neurons = 2
    output_neurons = 1

    # Random weights and biases
    W1 = np.random.randn(input_neurons, hidden_neurons)
    b1 = np.zeros((1, hidden_neurons))
    W2 = np.random.randn(hidden_neurons, output_neurons)
    b2 = np.zeros((1, output_neurons))

    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(X, W1) + b1
        hidden_output = sigmoid(hidden_input)
        final_input = np.dot(hidden_output, W2) + b2
        final_output = sigmoid(final_input)

        # Backward pass
        output_error = y - final_output
        output_delta = output_error * sigmoid_derivative(final_input)

        hidden_error = np.dot(output_delta, W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_input)

        # Update weights and biases
        W2 += learning_rate * np.dot(hidden_output.T, output_delta)
        b2 += learning_rate * np.sum(output_delta, axis=0, keepdims=True)
        W1 += learning_rate * np.dot(X.T, hidden_delta)
        b1 += learning_rate * np.sum(hidden_delta, axis=0, keepdims=True)

    return W1, b1, W2, b2

# Train the neural network
W1, b1, W2, b2 = train_xor_nn(X, y)

# Visualize XOR Decision Boundary
def visualize_xor_boundary(W1, b1, W2, b2):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Forward pass for grid points
    hidden_input = np.dot(grid, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    
    # Reshape predictions to grid shape
    predictions = final_output.reshape(xx.shape)
    
    # Plot
    plt.contourf(xx, yy, predictions, levels=[0, 0.5, 1], cmap="bwr", alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap="bwr", edgecolor="k")
    plt.title("XOR Decision Boundary")
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.show()

# Visualize the XOR decision boundary
visualize_xor_boundary(W1, b1, W2, b2)
