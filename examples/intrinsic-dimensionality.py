import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import pyEDAkit.IntrinsicDimensionality as eda_id
from generate_data import generate_1D_helix, generate_3D_helix, generate_scene

"""
--------- Scene ---------
Intrinsic Dimension Percentage Table:
   Intrinsic Dimension  Count  Percentage (%)
                   1.0   3881       64.683333
                   2.0   1144       19.066667
                   3.0    974       16.233333
                   4.0      1        0.016667
--------- 1D helix ---------
Pettis: 1.1188611882299895
CorrDim: 1.0519154515473388
MLE: 1.016437190647435
PackingNumbers: 0.9687679337468452
--------- 3D helix ---------
Pettis: 2.9694778990257538
CorrDim: 1.915960541356256
MLE: 2.2358864548083406
PackingNumbers: 1.4515438307243151

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
    dim = eda_id.MLE(neighbors)
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

idhat = eda_id.id_pettis(X)

print("Pettis:", idhat)

idhat = eda_id.corr_dim(X)

print("CorrDim:", idhat)

idhat = eda_id.MLE(X)

print("MLE:", idhat)

idhat = eda_id.packing_numbers(X)

print("PackingNumbers:", idhat)

print('--------- 3D helix ---------')
X, _ = generate_3D_helix(2000, 0.05, plot=True)

idhat = eda_id.id_pettis(X)

print("Pettis:", idhat)

idhat = eda_id.corr_dim(X)

print("CorrDim:", idhat)

idhat = eda_id.MLE(X)

print("MLE:", idhat)

idhat = eda_id.packing_numbers(X)

print("PackingNumbers:", idhat)
