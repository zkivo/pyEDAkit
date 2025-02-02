from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import connected_components
from matplotlib.colors import ListedColormap
import networkx as nx
from numpy.random import default_rng
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import seaborn as sns
import scipy.io
import pyEDAkit.clustering as eda_clustering
import pyEDAkit.linear as eda_lin
import pyEDAkit.standardization as eda_std
import numpy as np
import matplotlib.pyplot as plt

def test_documents_clustering_with_NMF():
    import numpy as np
    import matplotlib.pyplot as plt

    data = scipy.io.loadmat('datasets/nmfclustex.mat')
    X = data['nmfclustex']
    print(X.shape)

    words = [
        "human", "interface", "computer", "user", "system", 
        "response", "time", "EPS", "survey", "trees", "graph", "minors"
    ]

    document_titles = [
        "Human machine interface for Lab ABC computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user-perceived response time to error measurement",
        "The generation of random, binary, unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV: Widths of trees and well-quasi-ordering",
        "Graph minors: A survey"
    ]

    documents = [f"d{i}" for i in range(1, 10)]
    original_df = pd.DataFrame(data['nmfclustex'], index=words, columns=documents)

    print("Original Document Matrix:")
    print(original_df)

    X = X.astype(float)
    X = X / X.sum(axis=0)

    U, V = eda_lin.NMF(X, 2)

    # Normalize U and V using the formulas from the book
    U_normalized = U / np.sqrt(np.sum(U**2, axis=0))
    V_normalized = V * np.sqrt(np.sum(U**2, axis=0)[:, np.newaxis])

    print(V_normalized.T)

    idx = np.argmax(V_normalized.T, axis=1)  # Assign clusters based on the max in each row

    print(idx)

    v1, v2 = V_normalized[0, :], V_normalized[1, :]
    cluster_colors = ['blue', 'green']  # Colors for clusters 0 and 1

    # Create a scatter plot of the normalized V matrix with cluster-based colors
    plt.figure(figsize=(8, 6))
    for i, doc in enumerate(documents):
        cluster_color = cluster_colors[idx[i]]  # Color based on cluster assignment
        plt.scatter(v1[i], v2[i], marker='x', color=cluster_color, s=200, label=document_titles[i])
        plt.text(v1[i] + 0.02, v2[i] + 0.02, doc, fontsize=10, color=cluster_color)
    # Add plot details
    plt.title("Document Clustering Based on NMF")
    plt.xlabel("V1")
    plt.ylabel("V2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def test_spectral_clustering():
    # Generate datasets
    np.random.seed(0)  # For reproducibility
    mu1 = [2, 2]
    cov1 = np.eye(2)  # Identity matrix
    mu2 = [-1, -1]
    cov2 = [[1, 0.9], [0.9, 1]]

    X1 = np.random.multivariate_normal(mu1, cov1, 100)
    X2 = np.random.multivariate_normal(mu2, cov2, 100)

    # Concatenate datasets
    X = np.vstack((X1, X2))

    # Ground truth classes
    ground_truth = np.array([0] * 100 + [1] * 100)

    # Plot and save the ground truth
    plt.figure(figsize=(8, 6))
    plt.scatter(X1[:, 0], X1[:, 1], c='blue', marker='.', label='Class 1')
    plt.scatter(X2[:, 0], X2[:, 1], c='red', marker='+', label='Class 2')
    plt.title("Ground Truth Classes")
    plt.legend()
    plt.show()

    # Perform spectral clustering
    idx = eda_clustering.spectral(X, 2)

    # Plot spectral clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[idx == 0, 0], X[idx == 0, 1], c='green', marker='*', label='Cluster 1')
    plt.scatter(X[idx == 1, 0], X[idx == 1, 1], c='purple', marker='x', label='Cluster 2')
    plt.title("Spectral Clustering Results")
    plt.legend()
    plt.show()


def test_silhouette_as_book():
    # Load the Iris dataset
    data = load_iris()
    X = data.data

    # Perform KMeans clustering wtih 5 replication
    idx3, C3 = eda_clustering.kmeans(X, 3, init='random', n_init=5)
    idx4, C4 = eda_clustering.kmeans(X, 4, init='random', n_init=5)

    # The silhouette values are between -1 and 1
    # The higher the value, the better the clustering for that data point
    # Around 0 means that the data point is very close to the decision boundary
    # Below 0 means that the data point is closer to the neighboring cluster
    eda_clustering.silhouette(X, idx3, plot=True)
    eda_clustering.silhouette(X, idx4, plot=True)


def test_mojena_plot():
    # Load the yeast dataset
    file_path = 'datasets/lungB.mat'
    lungB = scipy.io.loadmat(file_path)
    X = lungB['lungB']

    X = X.T

    Z = eda_clustering.linkage(X, method='complete', metric='seuclidean')
    
    plt.figure(figsize=(12, 8))
    dendrogram(Z, truncate_mode='lastp', p=15, leaf_rotation=45, leaf_font_size=10)
    plt.title("Linkage Complete - seuclidean - LungB")
    plt.xlabel("Cluster Index")
    plt.ylim(40, 60)
    plt.ylabel("Distance")
    plt.show()

    # Plot the Mojena Rule graph
    eda_clustering.mojenaplot(Z, nc=10)

def test_cophenet_with_yeast():    
    file_path = 'datasets/yeast.mat'
    yeast = scipy.io.loadmat(file_path)

    X = yeast['data']
    D = pdist(X)

    # single linkage
    Z_single = eda_clustering.linkage(X, method='single')
    cophenet_single, d = eda_clustering.cophenet(Z_single, D)
    print('cophenet_single: ', cophenet_single)

    # complete linkage
    Z_complete = eda_clustering.linkage(X, method='complete')
    cophenet_complete, d = eda_clustering.cophenet(Z_complete, D)
    print('cophenet_complete: ', cophenet_complete)

def test_rand_index():
    # load iris labels
    iris = load_iris()
    X = iris.data
    y_true = iris.target

    # apply kmeans
    idx, C, = eda_clustering.kmeans(X, 3)

    print('rand_index: ', 
        eda_clustering.rand_index(y_true, idx, adjusted=False))
    print('(Adjusted) rand_index: ', 
        eda_clustering.rand_index(y_true, idx, adjusted=True))


########################################################
############## TEST LINKAGE FUNCTION ###################
########################################################
def test_linkage_with_yeast():
    file_path = 'datasets/yeast.mat'
    yeast = scipy.io.loadmat(file_path)

    X = yeast['data']

    print(X, type(X), X.shape)

    Z_single = eda_clustering.linkage(X, method='single')
    Z_complete = eda_clustering.linkage(X, method='complete')
    Z_average = eda_clustering.linkage(X, method='average')
    Z_centroid = eda_clustering.linkage(X, method='centroid')
    Z_ward = eda_clustering.linkage(X, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(Z_single, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Single Linkage")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.show()

    plt.figure(figsize=(12, 8))
    dendrogram(Z_complete, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Complete Linkage")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.show()

    plt.figure(figsize=(12, 8))
    dendrogram(Z_average, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Average Linkage")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.show()

    plt.figure(figsize=(12, 8))
    dendrogram(Z_centroid, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Centroid Linkage")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.show()

    plt.figure(figsize=(12, 8))
    dendrogram(Z_ward, truncate_mode='lastp', p=30, leaf_rotation=45, leaf_font_size=10)
    plt.title("Ward's Linkage")
    plt.xlabel("Cluster Index")
    plt.ylabel("Distance")
    plt.show()



def test_linkage_with_random():
    # Step 1: Randomly generate sample data with 20,000 observations
    np.random.seed(0)  # For reproducibility
    X = np.random.rand(20000, 3)

    # Step 2: Create a hierarchical cluster tree using the ward linkage method
    Z = eda_clustering.linkage(X, method='ward')

    # Step 3: Cluster the data into a maximum of four groups
    max_clusters = 4
    cluster_labels = eda_clustering.cluster(Z, 'MaxClust', max_clusters, criterion='maxclust')

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
    Z = eda_clustering.linkage(y, method='complete')

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
    Z = eda_clustering.linkage(X, method='ward')

    # 1) Cut off by distance = 0.7
    T_distance = eda_clustering.cluster(Z, 'Cutoff', 0.7, 'Criterion', 'distance')

    # 2) Cut off by inconsistent measure
    T_inconsist = eda_clustering.cluster(Z, 'Cutoff', 1.5)

    # 3) Force a maximum of 3 clusters
    T_maxclust = eda_clustering.cluster(Z, 'MaxClust', 3)

    # 4) Multiple cutoffs -> T is an m-by-l matrix
    T_multi = eda_clustering.cluster(Z, 'Cutoff', [0.7, 1.0, 1.5], 'Criterion', 'distance')
    print(T_multi.shape)  # (10, 3)


#############################################################
################# TEST KMEANS FUNCTION ######################
#############################################################
def test_kmeans_as_book():
    # Load the Iris dataset
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    data = eda_std.with_std_dev(df.iloc[:, :-1].to_numpy())

    # idx contains the cluster labels, C contains the cluster centers
    idx, C = eda_clustering.kmeans(data, 3)
    df['cluster'] = idx

    # Add cluster centers to the DataFrame for visualization
    scaler = StandardScaler()
    scaler.fit_transform(df.iloc[:, :-2])
    cluster_centers = scaler.inverse_transform(C)
    cluster_centers_df = pd.DataFrame(cluster_centers, columns=iris.feature_names)
    cluster_centers_df['cluster'] = ['Center 1', 'Center 2', 'Center 3']

    # Prepare data for the scatter matrix
    df_for_plot = pd.concat(
        [df, cluster_centers_df.assign(target=None)],
        ignore_index=True
    )

    # Plot the scatter matrix
    sns.pairplot(
        df_for_plot,
        vars=iris.feature_names,
        hue='cluster',
        markers=['o', 's', '^', 'X', 'D', '*'],  # Enough markers for all labels
        palette='tab10'
    )

    # Show the plot
    plt.suptitle('Scatter Matrix with KMeans Cluster Centers', y=1.02)
    plt.show()



def test_kmeans():
    # Sample data
    X = np.array([[1,2],[1,4],[1,0],
                  [10,2],[10,4],[10,0],
                  [5,2],[6,3],[7,4]])

    # 1) Basic call
    idx, C, sumd, D = eda_clustering.kmeans_matlab(X, 2)  # 2 clusters

    print("Cluster labels (idx):\n", idx)
    print("Centroids (C):\n", C)
    print("Within-cluster sums (sumd):\n", sumd)
    print("Distances to centroids (D):\n", D)

    # 2) With optional name-value arguments, e.g. 'Replicates'
    idx2, C2, sumd2, D2 = eda_clustering.kmeans_matlab(X, 3, 'Replicates', 5, 'MaxIter', 200, 'Display', 'iter')

    # Step 1: Load the Iris dataset
    iris_path = "datasets/iris_dataset.csv"
    iris_data = pd.read_csv(iris_path)

    # Use only petal_length and petal_width features (2D data)
    X = iris_data.iloc[:, [2, 3]].values  # Columns for petal_length and petal_width
    y_true = iris_data.iloc[:, -1].values  # True labels (species)

    # Map the target labels to numeric values for true labels
    label_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    y_numeric = np.array([label_mapping[label] for label in y_true])

    # Step 1: Apply k-means clustering using your custom function
    k = 3  # Number of clusters
    idx, C, sumd, D = eda_clustering.kmeans_matlab(X, k, 'Distance', 'sqeuclidean', 'Replicates', 5, 'MaxIter', 300)

    # Step 2: Create a 2D grid for the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))

    # Combine the grid into a list of points
    grid_points = np.c_[xx.ravel(), yy.ravel()]  # Flatten the grid into points for classification

    # Define the function to assign grid points to clusters
    def assign_to_clusters(grid_points, centroids):
        """
        Assign grid points to the nearest cluster based on centroids.
        """
        distances = np.linalg.norm(grid_points[:, None] - centroids, axis=2)  # Euclidean distance
        return np.argmin(distances, axis=1)  # Assign to the nearest centroid

    # Assign each grid point to a cluster
    Z = assign_to_clusters(grid_points, C)
    Z = Z.reshape(xx.shape)  # Reshape to match the grid dimensions

    # Step 4: Plot the results
    plt.figure(figsize=(10, 6))

    # Plot cluster regions
    cmap = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])  # Colors for regions
    plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.5)

    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=idx - 1, cmap='viridis', edgecolor='k', s=50, label='Data')

    # Plot centroids
    plt.scatter(C[:, 0], C[:, 1], c='red', s=200, marker='X', label='Centroids')

    # Add labels and title
    plt.title("K-means Clustering with Correctly Colored Areas")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Petal Width (cm)")
    plt.legend(loc='best')
    plt.show()

    # Step 5: Print results
    print("Cluster assignments (idx):")
    print(idx)
    print("\nCentroids (C):")
    print(C)
    print("\nWithin-cluster sum of distances (sumd):")
    print(sumd)


###################################################################
################# TEST MINIMUM SPANNING TREE ######################
###################################################################
def test_minspantree_as_book():
    # Generate random 2D points with clusters
    X = np.random.randn(100, 2)
    X[25:50, :] += 4
    X[50:75, :] -= 4
    X[75:100, 0] += 4
    X[75:100, 1] -= 4

    labels = np.concatenate((
            np.zeros(25),
            np.ones(25),
            2 * np.ones(25),
            3 * np.ones(25)
        ))

    # Scatter plot
    plt.figure(figsize=(8, 6))
    for i, label in enumerate(np.unique(labels)):
        plt.scatter(X[labels == label, 0], X[labels == label, 1], label=f"Cluster {int(label)}")
    plt.title("Input Data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

    idx = eda_clustering.mst_clustering(X, 4, plot=True)
    print('cluster labels: ', idx)
    


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
    T_prim, pred_prim = eda_clustering.minspantree(G)  # Assuming minspantree implements Prim's by default

    # Visualize MST with Prim's algorithm
    plt.subplot(1, 3, 2)
    nx.draw(T_prim, pos, with_labels=True, node_color='lightgreen', edge_color='blue', node_size=1000, font_size=10)
    labels = nx.get_edge_attributes(T_prim, 'weight')
    nx.draw_networkx_edge_labels(T_prim, pos, edge_labels=labels)
    plt.title("MST (Prim's Algorithm)")

    # 2) Compute MST using Kruskal's algorithm (Method='sparse')
    T_kruskal, pred_kruskal = eda_clustering.minspantree(G, 'Method', 'sparse', 'Root', 2, 'Type', 'forest')

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


def test_cophenet_with_random():
    # Step 1: Generate sample data
    np.random.seed(42)
    X = np.vstack([
        np.random.rand(5, 2),       # Cluster 1
        np.random.rand(5, 2) + 5   # Cluster 2 (shifted by +5)
    ])

    # Step 2: Compute pairwise distances
    Y = pdist(X)

    # Step 3: Perform hierarchical clustering with average linkage
    Z = eda_clustering.linkage(Y, method='average')

    # Step 4: Compute cophenetic correlation coefficient
    c, d = eda_clustering.cophenet(Z, Y)

    # Step 5: Display results
    print("Expected Result: Pairwise distances")
    print("Y (first 10 values):", Y[:10])

    print("\nActual Result: Cophenetic distances")
    print("d (first 10 values):", d[:10])

    print("\nCophenetic correlation coefficient:", c)

    # Step 6: Visualize clustering with dendrogram
    plt.figure(figsize=(10, 5))
    plt.title("Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    from scipy.cluster.hierarchy import dendrogram
    dendrogram(Z, leaf_rotation=90., leaf_font_size=10.)
    plt.show()


def test_silhouette():
    # Suppose we have data in X and cluster labels in "labels"
    X = np.random.rand(20, 2)
    labels = np.random.randint(0, 3, size=20)  # 3 clusters

    # 1) Use the default (Euclidean) distance and produce a plot
    s, fig = eda_clustering.silhouette_matlab(X, labels)
    print("Silhouette values:\n", s)

    # 2) Minkowski distance with exponent p=3, do not plot
    s2, _ = eda_clustering.silhouette_matlab(X, labels, Distance='minkowski', DistParameter={'p': 3}, do_plot=False)
    print("Silhouette values with Minkowski distance, p=3:\n", s2)


def test_eval_silhouette():
    rng = default_rng(42)
    n = 200
    X1 = rng.multivariate_normal(mean=[2, 2], cov=[[0.9, -0.0255], [-0.0255, 0.9]], size=n)
    X2 = rng.multivariate_normal(mean=[5, 5], cov=[[0.5, 0], [0, 0.3]], size=n)
    X3 = rng.multivariate_normal(mean=[-2, -2], cov=[[1, 0], [0, 0.9]], size=n)
    X = np.vstack([X1, X2, X3])

    # Evaluate the silhouette for k=1..6
    evaluation = eda_clustering.SilhouetteEvaluation(X,
                                      clusteringFunction='kmeans',
                                      KList=[1, 2, 3, 4, 5, 6],
                                      Distance='sqEuclidean',
                                      ClusterPriors='empirical')

    print("Inspected K:         ", evaluation.InspectedK)
    print("Criterion Values:    ", evaluation.CriterionValues)
    print("OptimalK:            ", evaluation.OptimalK)
    # Optional: plot the silhouette criterion
    evaluation.plot()

    # The best cluster solution's assignment:
    best_clusters = evaluation.OptimalY
    print("Shape of best cluster assignment:", best_clusters.shape)
    print("Some cluster labels:", np.unique(best_clusters[~np.isnan(best_clusters)]))


if __name__ == '__main__':
    # test_documents_clustering_with_NMF()
    # test_spectral_clustering()
    # test_rand_index()
    # test_linkage_with_yeast()
    # test_kmeans_as_book()
    # test_mojena_plot()
    test_minspantree_as_book()
    # test_cophenet_with_yeast()
    # test_silhouette_as_book()
    # test_kmeans()
    # test_minspantree()
    # test_cluster()
    # test_cophenet_with_random()
    # test_linkage_with_random()
    # test_silhouette()
    # test_eval_silhouette()
    pass