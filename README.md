# Exploratory Data Analysis (EDA) Python Toolkit
The project is sponsored by Malmö Universitet developed by Eng. Marco Schivo and Eng. Alberto Biscalchin under the supervision of Associete Professor Yuanji Cheng and is released under the MIT License. It is open source and available for anyone to use and contribute to.

Internal course Code reference: MA661E
## Overview
This repository implements MATLAB-style functions in Python for various data analysis, clustering, dimensionality reduction, and graph algorithms. These functions leverage popular Python libraries such as `numpy`, `scipy`, `matplotlib`, `networkx`, and `scikit-learn`, while maintaining a familiar MATLAB-like syntax and behavior.

---

## Features

### 1. **Clustering and Dimensionality Reduction**
- **Hierarchical Clustering** (`linkage`, `cluster`):
  - MATLAB-style hierarchical clustering using `scipy.cluster.hierarchy`.
  - Supports `single`, `complete`, `average`, `ward`, and other linkage methods.
  - MATLAB-like `cluster` function for cutting hierarchical clusters based on distance or cluster count.

- **K-Means Clustering** (`kmeans`):
  - MATLAB-style K-Means implementation with support for:
    - Initialization methods: `k-means++`, random, or user-specified.
    - Metrics: `sqeuclidean` distance.
    - Number of replicates and maximum iterations.
  - Outputs cluster assignments, centroids, and within-cluster sum of distances.

- **PCA and SVD**:
  - `PCA`: Computes Principal Components and visualizes scree plots and scatter matrices.
  - `SVD`: Performs Singular Value Decomposition with visualization of singular values.

- **Non-Negative Matrix Factorization (NMF)**:
  - Reduces data dimensionality while ensuring non-negativity constraints.

- **Factor Analysis (FA)**:
  - Estimates latent factors using sklearn's `FactorAnalysis`.

- **Linear Discriminant Analysis (LDA)**:
  - Reduces dimensionality while maximizing class separability.

- **Random Projection**:
  - Performs Gaussian Random Projection to reduce dimensionality.

---

### 2. **Graph Algorithms**
- **Minimum Spanning Tree (MST)** (`minspantree`):
  - MATLAB-style wrapper using `networkx`.
  - Supports both Prim's (`dense`) and Kruskal's (`sparse`) algorithms.
  - Option to extract a spanning tree for a specific connected component or spanning forest.

---

### 3. **Intrinsic Dimensionality Estimation**
- **Packing Numbers**:
  - Computes intrinsic dimensionality using packing arguments.

- **Maximum Likelihood Estimation (MLE)**:
  - Estimates intrinsic dimensionality using a k-Nearest Neighbor approach.

- **Correlation Dimension**:
  - Estimates intrinsic dimensionality via correlation methods.

- **Pettis Method**:
  - Computes intrinsic dimensionality using Pettis et al.'s algorithm.

---

### 4. **Normalization**
- **Standardization (`with_std_dev`)**:
  - Standardizes data with zero-mean and unit variance.
  
- **Min-Max Normalization (`min_max_norm`)**:
  - Rescales data to a range between 0 and 1.

- **Sphering (`sphering`)**:
  - Whitens data, decorrelating variables and setting variance to 1.


---
# Examples
## Clustering
### 1. **`Linkage` function**
```python
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
    leaf_font_size=10      # Adjust font size for clarity
)
plt.title('Dendrogram')
plt.xlabel('Leaf Nodes')
plt.ylabel('Linkage Distance')
plt.show()
```


## Dependencies
This repository requires the following Python libraries:
- `numpy>=1.21.0`
- `scipy>=1.7.0`
- `matplotlib>=3.4.0`
- `seaborn>=0.11.0`
- `scikit-learn>=0.24.0`
- `networkx>=2.5`
- `pandas>=1.3.0`


---

## Contributing
Feel free to fork this repository, report issues, or contribute by adding new MATLAB-style functions or improving existing ones. 

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.

