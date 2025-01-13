import numpy as np
from pyEDAkit.IntrinsicDimensionality import id_pettis
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

print("Estimated intrinsic dimensionality:", idhat)
