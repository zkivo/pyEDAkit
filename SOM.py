import numpy as np
import scipy.io
import pandas as pd
from matplotlib.patches import RegularPolygon
from minisom import MiniSom
import matplotlib.pyplot as plt

# Load the MATLAB data
mat_data = scipy.io.loadmat('data/oronsay.mat')

# Extract data and labels
data = mat_data['oronsay']
midden = mat_data['midden']

# normalize data
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Flatten and extract strings from midden
midden = [item for item in midden.ravel()]

# Initialize and train the SOM
som_x, som_y = 10, 10  # Define SOM grid size
som = MiniSom(som_x, som_y, data.shape[1], sigma=1.0, learning_rate=0.5, topology='hexagonal', neighborhood_function='gaussian')
som.random_weights_init(data)
som.train_random(data, 100000)  # Number of iterations

# Compute the U-Matrix
u_matrix = np.zeros((som_x, som_y))
for x in range(som_x):
    for y in range(som_y):
        neighbors = [
            (x + dx, y + dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            if (0 <= x + dx < som_x and 0 <= y + dy < som_y) and not (dx == 0 and dy == 0)
        ]
        distances = [
            np.linalg.norm(som._weights[x, y] - som._weights[nx, ny])
            for nx, ny in neighbors
        ]
        u_matrix[x, y] = np.mean(distances)

# Plot the U-Matrix
plt.figure(figsize=(10, 10))
plt.title("U-Matrix of the SOM", fontsize=16)
plt.imshow(u_matrix.T, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Distance')
plt.xticks([])
plt.yticks([])

# Overlay class points on the SOM
for idx, x in enumerate(data):
    winner = som.winner(x)
    plt.text(winner[0], winner[1], str(midden[idx]), color='black', fontsize=8, ha='center', va='center')

plt.show()