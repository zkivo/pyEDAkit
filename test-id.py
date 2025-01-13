import numpy as np
import matplotlib.pyplot as plt
from pyEDAkit.IntrinsicDimensionality import id_pettis, corr_dim, MLE, \
    packing_numbers
from generate_data import generate_1D_helix, generate_3D_helix, generate_scene
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import pandas as tools

"""
Results:
    --------- 1D helix ---------
    Pettis: 1.1047889445671362
    CorrDim: 1.0309670763066818
    MLE: 1.014824818196277
    PackingNumbers: 0.944554645367179

    --------- 3D helix ---------
    Pettis: 2.977299222123266
    CorrDim: 1.8408523277805207
    MLE: 2.245914166730645
    PackingNumbers: 1.4337496475758493
"""

print('--------- Scene ---------')
X = generate_scene(plot=False)

# Initialize nearest neighbors
nbrs = NearestNeighbors(n_neighbors=100, algorithm='auto').fit(X)
_, indices = nbrs.kneighbors(X)

# Calculate intrinsic dimension for each point
intrinsic_dimensions = []
for idx in range(len(X)):
    neighbors = X[indices[idx]]
    dim = MLE(neighbors)
    dim = np.round(dim)
    intrinsic_dimensions.append(dim)

# Replace values > 3 with 4
intrinsic_dimensions = np.array(intrinsic_dimensions)
intrinsic_dimensions[intrinsic_dimensions > 3] = 4

# Calculate percentage of each intrinsic dimension
unique, counts = np.unique(intrinsic_dimensions, return_counts=True)
percentages = (counts / len(intrinsic_dimensions)) * 100
percentage_table = pd.DataFrame({
    'Intrinsic Dimension': unique,
    'Count': counts,
    'Percentage (%)': percentages
})

# Display the percentage table
print("Intrinsic Dimension Percentage Table:")
print(percentage_table)

# Plot the 3D graph with intrinsic dimensions as colors
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=intrinsic_dimensions, cmap='viridis', s=10)
ax.set_title('3D Scatter Plot Colored by Local Intrinsic Dimension (k=100)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
colorbar = fig.colorbar(scatter, ax=ax, label='Intrinsic Dimension')
plt.show()


print('--------- 1D helix ---------')
X = generate_1D_helix(500, plot=True)

idhat = id_pettis(X)

print("Pettis:", idhat)

idhat = corr_dim(X)

print("CorrDim:", idhat)

idhat = MLE(X)

print("MLE:", idhat)

idhat = packing_numbers(X)

print("PackingNumbers:", idhat)

print('--------- 3D helix ---------')
X, _ = generate_3D_helix(2000, 0.05, plot=True)

idhat = id_pettis(X)

print("Pettis:", idhat)

idhat = corr_dim(X)

print("CorrDim:", idhat)

idhat = MLE(X)

print("MLE:", idhat)

idhat = packing_numbers(X)

print("PackingNumbers:", idhat)
