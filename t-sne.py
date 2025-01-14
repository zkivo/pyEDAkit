import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

# Define the function
def generate_and_visualize_clusters():
    # Generate synthetic 3D data
    centers = [[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]]  # Cluster centers
    cluster_std = 0.5  # Standard deviation of the clusters
    X, y = make_blobs(n_samples=400, centers=centers, cluster_std=cluster_std, random_state=42)

    # Plot the 3D clusters
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection='3d')
    colors = plt.cm.tab10(np.linspace(0, 1, len(centers)))

    for cluster_idx in np.unique(y):
        cluster_points = X[y == cluster_idx]
        ax.scatter(
            cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
            label=f'Cluster {cluster_idx}', color=colors[cluster_idx]
        )

    ax.set_title("3D Clusters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # Apply T-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_2d = tsne.fit_transform(X)

    # Plot the 2D clusters
    ax2 = fig.add_subplot(122)
    for cluster_idx in np.unique(y):
        cluster_points_2d = X_2d[y == cluster_idx]
        ax2.scatter(
            cluster_points_2d[:, 0], cluster_points_2d[:, 1],
            label=f'Cluster {cluster_idx}', color=colors[cluster_idx]
        )

    ax2.set_title("2D Clusters (T-SNE)")
    ax2.set_xlabel("TSNE Component 1")
    ax2.set_ylabel("TSNE Component 2")
    ax2.legend()

    plt.tight_layout()
    plt.show()

# Call the function
generate_and_visualize_clusters()
