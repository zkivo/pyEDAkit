import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import fcluster
from pyEDAkit.clustering import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram

# Assuming your linkage function is already defined as `linkage`

# Step 1: Randomly generate sample data with 20,000 observations
np.random.seed(0)  # For reproducibility
X = np.random.rand(20000, 3)

# Step 2: Create a hierarchical cluster tree using the ward linkage method
Z = linkage(X, method='ward')

# Step 3: Cluster the data into a maximum of four groups
max_clusters = 4
cluster_labels = fcluster(Z, max_clusters, criterion='maxclust')

# Step 4: Plot the result in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_labels, cmap='viridis', s=10)
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

plt.title('3D Scatter Plot of Hierarchical Clustering')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
plt.show()

# Step 5: Define the dissimilarity matrix
X = np.array([
    [0, 1, 2, 3],
    [1, 0, 4, 5],
    [2, 4, 0, 6],
    [3, 5, 6, 0]
])

# Step 6: Convert the dissimilarity matrix to vector form using squareform
y = squareform(X)

# Step 7: Create a hierarchical cluster tree using the 'complete' method
Z = linkage(y, method='complete')

# Step 8: Print the resulting linkage matrix
print("Linkage matrix (Z):")
print(Z)

# Step 9: Plot the dendrogram
plt.figure(figsize=(10, 6))
dendrogram(
    Z,
    labels=[1, 2, 3, 4],  # Use MATLAB-style indices for the leaf nodes
    leaf_font_size=10,      # Adjust font size for clarity
    reorder=[0, 1, 2, 3]   # Reorder leaves to appear as 1-2-3-4
)
plt.title('Dendrogram')
plt.xlabel('Leaf Nodes')
plt.ylabel('Linkage Distance')
plt.show()
