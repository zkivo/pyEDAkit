import numpy as np
from pyEDAkit.IntrinsicDimensionality import id_pettis, corr_dim, MLE, \
    packing_numbers
import matplotlib.pyplot as plt

n = 500

# Generate random theta values
theta = np.random.uniform(0, 4 * np.pi, n)

# Compute x, y, z coordinates for the helix
x = np.cos(theta)
y = np.sin(theta)
z = 0.1 * theta

# Combine into a data matrix
X = np.column_stack((x, y, z))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()

# Use id_pettis to estimate intrinsic dimensionality
idhat = id_pettis(X)

print("Pettis:", idhat)

idhat = corr_dim(X)

print("CorrDim:", idhat)

idhat = MLE(X)

print("MLE:", idhat)

idhat = packing_numbers(X)

print("PackingNumbers:", idhat)
