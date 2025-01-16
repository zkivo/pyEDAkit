from scipy.cluster.hierarchy import fcluster
from pyEDAkit.clustering import linkage, cluster, kmeans
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
from pyEDAkit.clustering import kmeans
import networkx as nx
from pyEDAkit.clustering import minspantree

########################################################
############## TEST LINKAGE FUNCTION ###################
########################################################
def test_linkage():
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


#############################################################
################# TEST CLUSTER FUNCTION #####################
#############################################################
def test_cluster():
    # Generate sample data
    X = np.random.rand(10, 3)

    # Compute linkage matrix
    Z = linkage(X, method='ward')

    # 1) Cut off by distance = 0.7
    T_distance = cluster(Z, 'Cutoff', 0.7, 'Criterion', 'distance')

    # 2) Cut off by inconsistent measure
    T_inconsist = cluster(Z, 'Cutoff', 1.5)

    # 3) Force a maximum of 3 clusters
    T_maxclust = cluster(Z, 'MaxClust', 3)

    # 4) Multiple cutoffs -> T is an m-by-l matrix
    T_multi = cluster(Z, 'Cutoff', [0.7, 1.0, 1.5], 'Criterion', 'distance')
    print(T_multi.shape)  # (10, 3)


#############################################################
################# TEST KMEANS FUNCTION ######################
#############################################################
def test_kmeans():
    # Sample data
    X = np.array([[1,2],[1,4],[1,0],
                  [10,2],[10,4],[10,0],
                  [5,2],[6,3],[7,4]])

    # 1) Basic call
    idx, C, sumd, D = kmeans(X, 2)  # 2 clusters

    print("Cluster labels (idx):\n", idx)
    print("Centroids (C):\n", C)
    print("Within-cluster sums (sumd):\n", sumd)
    print("Distances to centroids (D):\n", D)

    # 2) With optional name-value arguments, e.g. 'Replicates'
    idx2, C2, sumd2, D2 = kmeans(X, 3, 'Replicates', 5, 'MaxIter', 200, 'Display', 'iter')



    # Step 1: Load the Iris dataset
    iris_path = "../datasets/iris_dataset.csv"
    iris_data = pd.read_csv(iris_path)

    # Extract features and target labels
    X = iris_data.iloc[:, :-1].values  # First 4 columns (features)
    y_true = iris_data.iloc[:, -1].values  # Last column (true labels)

    # Map the target labels to numeric values
    label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y_numeric = np.array([label_mapping[label] for label in y_true])

    # Step 2: Apply K-means clustering
    k = 3  # Number of clusters (as Iris dataset has 3 classes)
    idx, C, sumd, D = kmeans(X, k, 'Distance', 'sqeuclidean', 'Replicates', 5, 'MaxIter', 300)

    # Step 3: Reduce dimensionality for visualization (using PCA)
    pca = PCA(n_components=2)  # Reduce to 2D
    X_pca = pca.fit_transform(X)

    # Transform centroids to PCA space
    C_pca = pca.transform(C)

    # Step 4: Visualize the clustering results with colored areas
    plt.figure(figsize=(12, 6))

    # Plot the true labels
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='viridis', edgecolor='k', s=50)
    plt.title("True Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter1, label="Class")

    # Plot the K-means cluster assignments with colored areas
    plt.subplot(1, 2, 2)

    # Create a grid to color the background
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # Predict the cluster for each point in the grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = kmeans(pca.inverse_transform(grid_points), k, 'Distance', 'sqeuclidean')[0]
    Z = Z.reshape(xx.shape)

    # Plot the filled contour for the clusters
    cmap = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.4)

    # Scatter the points
    scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=idx, cmap='viridis', edgecolor='k', s=50)
    plt.scatter(C_pca[:, 0], C_pca[:, 1], c='red', s=200, marker='X', label="Centroids")  # Mark centroids
    plt.title("K-means Clustering with Colored Areas")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()
    plt.colorbar(scatter2, label="Cluster")

    plt.tight_layout()
    plt.show()

    # Step 5: Calculate and print accuracy
    # Map clusters to the closest true labels to calculate accuracy
    from scipy.stats import mode

    # Remap clusters to best match true labels
    remapped_idx = np.zeros_like(idx)
    for cluster in range(1, k + 1):  # Clusters are 1-based
        mask = (idx == cluster)
        remapped_idx[mask] = mode(y_numeric[mask])[0]

    # Calculate accuracy
    accuracy = accuracy_score(y_numeric, remapped_idx)
    print(f"Clustering Accuracy: {accuracy:.2f}")

    # Print results
    print("Cluster assignments (idx):")
    print(idx)
    print("\nCentroids (C):")
    print(C)
    print("\nWithin-cluster sum of distances (sumd):")
    print(sumd)


###################################################################
################# TEST MINIMUM SPANNING TREE ######################
###################################################################
def test_minspantree():
    # Create a graph with weighted edges
    G = nx.Graph()
    G.add_weighted_edges_from([
        (1, 2, 2.0),
        (2, 3, 1.5),
        (2, 4, 3.0),
        (1, 5, 4.0),
        (3, 5, 2.5),
        (4, 5, 1.0),
        (5, 6, 2.0)
    ])

    # Visualize the original graph
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    pos = nx.spring_layout(G, seed=42)  # For consistent layout
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Original Graph")

    # 1) Compute MST using Prim's algorithm (default method)
    T_prim, pred_prim = minspantree(G)  # Assuming minspantree implements Prim's by default

    # Visualize MST with Prim's algorithm
    plt.subplot(1, 3, 2)
    nx.draw(T_prim, pos, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=1000, font_size=10)
    labels = nx.get_edge_attributes(T_prim, 'weight')
    nx.draw_networkx_edge_labels(T_prim, pos, edge_labels=labels)
    plt.title("MST (Prim's Algorithm)")

    # 2) Compute MST using Kruskal's algorithm (Method='sparse')
    T_kruskal, pred_kruskal = minspantree(G, 'Method', 'sparse', 'Root', 2, 'Type', 'forest')

    # Visualize MST with Kruskal's algorithm
    plt.subplot(1, 3, 3)
    nx.draw(T_kruskal, pos, with_labels=True, node_color='lightcoral', edge_color='purple', node_size=1000, font_size=10)
    labels = nx.get_edge_attributes(T_kruskal, 'weight')
    nx.draw_networkx_edge_labels(T_kruskal, pos, edge_labels=labels)
    plt.title("MST (Kruskal's Algorithm)")

    plt.tight_layout()
    plt.show()

    # Print details of the MSTs
    print("\n--- MST using Prim's Algorithm ---")
    print("Edges of T (Prim):", list(T_prim.edges(data=True)))
    print("Predecessors (Prim):", pred_prim)

    print("\n--- MST using Kruskal's Algorithm ---")
    print("Edges of T (Kruskal):", list(T_kruskal.edges(data=True)))
    print("Predecessors (Kruskal):", pred_kruskal)


if __name__ == '__main__':
    test_linkage()
    test_cluster()
    test_kmeans()
    test_minspantree()