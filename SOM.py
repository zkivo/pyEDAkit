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
som = MiniSom(som_x, som_y, data.shape[1], sigma=1.0, learning_rate=0.5, topology='rectangular', neighborhood_function='gaussian')
som.random_weights_init(data)
som.train_random(data, 1000000, verbose=True)  # Number of iterations

u_matrix = som.distance_map()

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