import numpy as  np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ------- Frame -------

def make_data(n_sample=100):
    rnd = np.random.RandomState(42)
    x = rnd.uniform(1,7,size=n_sample)
    y_no_noise = (np.sin(4*x) + x + 5)
    y = (y_no_noise + rnd.normal(size=len(x))/2)
    y *= 5
    return x.reshape(-1,1), y.reshape(-1,1)

x,y = make_data()

print(y.shape)
print(x.shape)
# Both (100, 1)

plt.scatter(x,y)
plt.title("Scatter Plot")
plt.xlabel("Advertising Expediture (million dollar)")
plt.ylabel("Sales Revenue (million dollar)")
plt.show()

# ------- Model; Define Cost Function -------

#  J(θ) = 1/2m ∑mi=1 (hθ(x^(i))−y^(i)^)2

# m is number of training example
m = len(y)
print(m)
# expect 100

# Adding ones into X
ones = np.ones((m, 1))
x = np.append(ones, x, axis = 1)
print(x.shape)
# expect (100,2)

# Model parameter
theta_initial = np.zeros((2, 1))
print(theta_initial.shape)
# expect (2, 1)

# Model hypothesis: h = theta_0 * x_0 + theta_1 * x_1
h = x.dot(theta_initial)
print(h.shape)
# expect (100, 1)

# Initialize Hyperparameters
iters = 300
lr = 0.1

# Plot model predictions to chart
plt.scatter(x[:,1], y)
plt.plot(x[:,1], x.dot(theta_initial).reshape(100,), 'r--')
plt.vlines(x[:,1], y, x.dot(theta_initial), colors='k', linestyles='dotted')

plt.title("Scatter Plot")
plt.xlabel("Advertising Expediture (million dollar)")
plt.ylabel("Sales Revenue (million dollar)")

# Train Model (Algorithm)
def cost_func(x, y, theta):
    m = len(y)
    h = x.dot(theta)
    return np.sum((h - y) ** 2) / (2 * m)

def cost_func_non_vectorized(X, y, theta):
    m = len(y)
    total_cost = 0
    for i in range(m):
        h_i = X[i].dot(theta)
        error_i = h_i - y[i]
        total_cost += error_i ** 2
    return total_cost / (2 * m)

# ----- Visualise the cost function -------

# Define range for theta_0 and theta_1
theta_0_vals = np.linspace(0, 30, 100)
theta_1_vals = np.linspace(0, 4, 100)

# Create meshgrid
Theta_0, Theta_1 = np.meshgrid(theta_0_vals, theta_1_vals)

# Calculate cost for each theta_0, theta_1 pair
J_vals = np.zeros_like(Theta_0)
for i in range(Theta_0.shape[0]):
    for j in range(Theta_0.shape[1]):
        theta = np.array([[Theta_0[i, j]], [Theta_1[i, j]]])
        J_vals[i, j] = cost_func(x, y, theta)

# Plot the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Theta_0, Theta_1, J_vals, cmap='turbo')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
plt.title('Cost Function J(theta_0, theta_1)')
# Set the viewing angle (elevation, azimuthal)
ax.view_init(elev=15, azim=-60)
plt.show()

# Find the indices of the maximum value in J_vals
max_idx = np.unravel_index(np.argmax(J_vals), J_vals.shape)

# Retrieve the corresponding theta_0 and theta_1 values
theta_0_max = Theta_0[max_idx]
theta_1_max = Theta_1[max_idx]

# Print the results
print("Theta values leading to maximum cost:")
print("Theta 0:", theta_0_max)
print("Theta 1:", theta_1_max)

# ----- Gradient Descent -----

def gradient_descent(x, y, theta, lr, iters):
    m = len(y)
    cost_history = []

    for i in range(iters):
        h = x.dot(theta)
        gradient = (1 / m) * x.T.dot(h - y)
        theta -= lr * gradient
        cost_history.append(cost_func(x, y, theta))

    return theta, cost_history

theta_final, cost_history = gradient_descent(x, y, theta_initial, lr, iters)

# Apply trained Model

plt.scatter(x[:,1], y)
plt.plot(x[:,1], x.dot(theta_final).reshape(-1,), 'g-', label='Trained Model')
plt.title("Final Model Fit")
plt.legend()
plt.show()
