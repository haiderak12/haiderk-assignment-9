import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # learning rate
        self.activation_fn = activation  # activation function
        # Define layers and initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))
        # For storing activations and gradients
        self.cache = {}
        self.grads = {}

    def forward(self, X):
        # Forward pass, apply layers to input X
        z1 = np.dot(X, self.W1) + self.b1
        if self.activation_fn == 'tanh':
            a1 = np.tanh(z1)
        elif self.activation_fn == 'relu':
            a1 = np.maximum(0, z1)
        elif self.activation_fn == 'sigmoid':
            a1 = 1 / (1 + np.exp(-z1))
        else:
            raise ValueError('Unsupported activation function')

        z2 = np.dot(a1, self.W2) + self.b2
        y_hat = 1 / (1 + np.exp(-z2))  # Sigmoid activation for binary classification

        # Store activations for visualization
        self.cache['X'] = X
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['y_hat'] = y_hat

        return y_hat

    def backward(self, X, y):
        m = y.shape[0]
        # Retrieve stored activations
        a1 = self.cache['a1']
        y_hat = self.cache['y_hat']
        X = self.cache['X']

        # Map labels from -1 and +1 to 0 and 1
        y_01 = (y + 1) / 2

        # Compute gradients using chain rule
        dz2 = y_hat - y_01  # Gradient of loss with respect to z2

        # Gradients for W2 and b2
        dW2 = np.dot(a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Backpropagate to hidden layer
        da1 = np.dot(dz2, self.W2.T)
        if self.activation_fn == 'tanh':
            dz1 = da1 * (1 - np.square(a1))
        elif self.activation_fn == 'relu':
            dz1 = da1 * (a1 > 0)
        elif self.activation_fn == 'sigmoid':
            dz1 = da1 * a1 * (1 - a1)
        else:
            raise ValueError('Unsupported activation function')

        # Gradients for W1 and b1
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights with gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # Store gradients for visualization
        self.grads['dW1'] = dW1
        self.grads['db1'] = db1
        self.grads['dW2'] = dW2
        self.grads['db2'] = db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps by calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden features
    hidden_features = mlp.cache['a1']
    y_colors = (y + 1) / 2  # Map labels to 0 and 1 for colors
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y_colors.ravel(), cmap='bwr', alpha=0.7)

    # Hyperplane visualization in the hidden space
    W2 = mlp.W2.reshape(-1)
    b2 = mlp.b2.reshape(-1)
    xx, yy = np.meshgrid(np.linspace(hidden_features[:, 0].min(), hidden_features[:, 0].max(), 10),
                         np.linspace(hidden_features[:, 1].min(), hidden_features[:, 1].max(), 10))
    if W2[2] != 0:
        zz = (-W2[0]*xx - W2[1]*yy - b2)/W2[2]
        ax_hidden.plot_surface(xx, yy, zz, alpha=0.3)
    ax_hidden.set_xlabel('Hidden Unit 1')
    ax_hidden.set_ylabel('Hidden Unit 2')
    ax_hidden.set_zlabel('Hidden Unit 3')
    ax_hidden.set_title('Hidden Layer Representation')

    # Plot input layer decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx_input, yy_input = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_input = np.c_[xx_input.ravel(), yy_input.ravel()]
    # Forward pass for grid points
    z1_grid = np.dot(grid_input, mlp.W1) + mlp.b1
    if mlp.activation_fn == 'tanh':
        a1_grid = np.tanh(z1_grid)
    elif mlp.activation_fn == 'relu':
        a1_grid = np.maximum(0, z1_grid)
    elif mlp.activation_fn == 'sigmoid':
        a1_grid = 1 / (1 + np.exp(-z1_grid))
    z2_grid = np.dot(a1_grid, mlp.W2) + mlp.b2
    y_hat_grid = 1 / (1 + np.exp(-z2_grid))
    y_hat_grid = y_hat_grid.reshape(xx_input.shape)
    # Plot decision boundary
    ax_input.contourf(xx_input, yy_input, y_hat_grid, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax_input.scatter(X[:, 0], X[:, 1], c=y_colors.ravel(), cmap='bwr', edgecolor='k')
    ax_input.set_title('Input Space Decision Boundary')

    # Visualize features and gradients as circles and edges
    ax_gradient.set_xlim(-0.1, 2.1)
    ax_gradient.set_ylim(-0.1, 1.1)
    # Positions for the nodes
    input_layer_x = [0, 0]
    input_layer_y = [0.3, 0.7]
    hidden_layer_x = [1, 1, 1]
    hidden_layer_y = [0.2, 0.5, 0.8]
    output_layer_x = [2]
    output_layer_y = [0.5]
    # Plot nodes
    for x, y in zip(input_layer_x, input_layer_y):
        circle = Circle((x, y), radius=0.05, fill=True, color='blue')
        ax_gradient.add_patch(circle)
    for x, y in zip(hidden_layer_x, hidden_layer_y):
        circle = Circle((x, y), radius=0.05, fill=True, color='green')
        ax_gradient.add_patch(circle)
    for x, y in zip(output_layer_x, output_layer_y):
        circle = Circle((x, y), radius=0.05, fill=True, color='red')
        ax_gradient.add_patch(circle)
    # Edges from input to hidden layer
    for i, (x1, y1) in enumerate(zip(input_layer_x, input_layer_y)):
        for j, (x2, y2) in enumerate(zip(hidden_layer_x, hidden_layer_y)):
            grad = mlp.grads['dW1'][i, j]
            linewidth = np.abs(grad) * 1000  # Scale gradient magnitude for visualization
            ax_gradient.plot([x1, x2], [y1, y2], 'k-', linewidth=linewidth)
    # Edges from hidden to output layer
    for i, (x1, y1) in enumerate(zip(hidden_layer_x, hidden_layer_y)):
        x2, y2 = output_layer_x[0], output_layer_y[0]
        grad = mlp.grads['dW2'][i, 0]
        linewidth = np.abs(grad) * 1000  # Scale gradient magnitude for visualization
        ax_gradient.plot([x1, x2], [y1, y2], 'k-', linewidth=linewidth)
    ax_gradient.axis('off')
    ax_gradient.set_title('Network Gradients')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
