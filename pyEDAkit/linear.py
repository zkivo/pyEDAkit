import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import NMF as skNMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.random_projection import GaussianRandomProjection

def PCA(X, d, covariance = True, plot = False):
    X_mean = X.mean(axis=0)
    X = X - X_mean
    S = None
    if covariance:
        S = np.cov(X, rowvar=False)
    else:
        S = np.corrcoef(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(S)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    Z = (X @ sorted_eigenvectors)[:, :d]

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(sorted_eigenvalue) + 1), sorted_eigenvalue, marker='o', linestyle='-')
        plt.plot(d, sorted_eigenvalue[d - 1], 'ro', label = 'd')
        plt.title('Scree Plot')
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue Magnitude')
        plt.legend()
        plt.grid(True)
        # scatter matrix of Z
        sns.pairplot(pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(d)]), diag_kind='kde')
        plt.legend()
        plt.show()

    return Z

def SVD(X, plot = False):
    # It provides a way to find the PCs without explicitly calculating 
    #   the covariance matrix.
    # The plot of the singular values is similar to the scree plot in PCA.
    U, S, Vt = np.linalg.svd(X)
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(S) + 1), S, marker='o', linestyle='-', label='Singular Values')
        plt.title('Singular Value Decomposition')
        plt.xlabel('Index')
        plt.ylabel('Singular Value')
        plt.grid(True)
        plt.legend()
        plt.show()

    return U, S, Vt

def NMF(X, d, plot = False):
    # is X non-negative?
    if np.any(X < 0):
        print('Error: X contains negative values.')
        return None

    nmf_model = skNMF(n_components=d, init='random', random_state=42) 
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, H.shape[0] + 1), np.linalg.norm(H, axis=1), marker='o', linestyle='-', label='Component Norms')
        plt.title('NMF Component Contributions')
        plt.xlabel('Component Index')
        plt.ylabel('Norm of Component')
        plt.grid(True)
        plt.legend()
        plt.show()

    return W, H

def FA(X, d, plot = False):
    fa = FactorAnalysis(n_components=d)
    Z = fa.fit_transform(X)

    factor_df = pd.DataFrame(Z, columns=[f"Factor {i+1}" for i in range(d)])

    if plot:
        sns.pairplot(factor_df, diag_kind="kde")
        plt.suptitle("Scatter Plots of Factors", y=1.02)
        plt.show()

    return Z

def LDA(X, y, plot = False):
    lda = LinearDiscriminantAnalysis(n_components=1)  # Reduce to 1 dimensions
    Z = lda.fit_transform(X, y)

    # Create a DataFrame for easy visualization
    lda_df = pd.DataFrame({"LD1": Z.flatten(), "Class": y})

    if plot:
        unique_classes = np.unique(y)
        palette = sns.color_palette("Set2", len(unique_classes))
        class_colors = {class_id: palette[i] for i, class_id in enumerate(unique_classes)}

        plt.figure(figsize=(8, 6))
        for class_id, color in class_colors.items():
            sns.kdeplot(
                lda_df.loc[lda_df["Class"] == class_id, "LD1"],
                color=color,
                fill=True,
                alpha=0.6,
                label=f"Class {class_id}"
            )
        # Scatter points on the x-axis for each class
        for class_id, color in class_colors.items():
            class_points = lda_df[lda_df["Class"] == class_id]["LD1"]
            plt.scatter(
                class_points,
                [-0.01] * len(class_points),  # Slightly below the x-axis for separation
                color=color,
                alpha=0.7
            )
        plt.title("LDA with KDE")
        plt.xlabel("Linear Discriminant")
        plt.ylabel("Density")
        plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        plt.grid()
        plt.show()

    return Z

def RandProj(X, d, plot = False):
    rp = GaussianRandomProjection(n_components=d, random_state=42)
    Z = rp.fit_transform(X)

    # Convert reduced data to a DataFrame
    reduced_feature_names = [f'Component_{i+1}' for i in range(d)]
    reduced_data_df = pd.DataFrame(Z, columns=reduced_feature_names)

    if plot:
        sns.pairplot(reduced_data_df, diag_kind='kde', corner=True)
        plt.suptitle('Random projection')
        plt.show()