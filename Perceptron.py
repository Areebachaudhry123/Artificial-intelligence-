import numpy as np
import matplotlib.pyplot as plt

# Define the dataset (X: inputs, y: outputs)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Example of linearly separable data

def perceptron(X, y, learning_rate, epochs):
    # Initialize weights and bias
    weights = np.zeros(X.shape[1])
    bias = 0

    for epoch in range(epochs):
        # Compute predictions
        linear_output = np.dot(X, weights) + bias
        predicted = (linear_output > 0).astype(int)  # Faster than np.where
        errors = y - predicted

        # Update weights and bias if there are errors
        if np.any(errors):  # Simplified condition
            weights += learning_rate * np.dot(errors, X)
            bias += learning_rate * np.sum(errors)
        else:
            print(f"Training converged after {epoch + 1} epochs.")
            break

    return weights, bias

def predict(X, weights, bias):
    """Predict the output using the trained weights and bias."""
    return (np.dot(X, weights) + bias > 0).astype(int)

def plot_perceptron_boundary(X, y, weights, bias):
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=50)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Decision Boundary")

    # Plot decision boundary
    x_boundary = np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 100)
    y_boundary = -(weights[0] * x_boundary + bias) / (weights[1] + 1e-10)  # Numerical stability
    plt.plot(x_boundary, y_boundary, color='green')
    plt.show()

# Train the perceptron
weights, bias = perceptron(X, y, learning_rate=0.1, epochs=10)

# Visualize the decision boundary
plot_perceptron_boundary(X, y, weights, bias)

# Test the trained perceptron with new data
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 1]])
predictions = predict(test_data, weights, bias)

# Displaying predictions for the new data
print(f"Predictions for the test data:\n{test_data}")
print(f"Predictions: {predictions}")
