from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pyEDAkit import standardization as eda_std


def scatter_plot(x, y, targets, title='Title', class_names=None):
    """
    Make a scatter plot where each unique integer in `targets` is plotted
    separately, with a legend entry.

    Parameters
    ----------
    x, y : 1D arrays of the same length
    targets : 1D array of class labels (integers)
    title : str
        Title of the figure.
    class_names : list of str, optional
        Names to appear in the legend. If None, just use numeric labels.
    """
    plt.figure()
    unique_labels = np.unique(targets)

    for lbl in unique_labels:
        mask = (targets == lbl)
        if class_names is not None and 0 <= lbl < len(class_names):
            label_str = class_names[lbl]
        else:
            label_str = f"Class {lbl}"
        plt.scatter(x[mask], y[mask], s=10, label=label_str)

    plt.axhline(0, color='gray', linestyle='--')  # Horizontal dotted line at y=0
    plt.axvline(0, color='gray', linestyle='--')  # Vertical dotted line at x=0
    plt.title(title)
    plt.legend()
    plt.draw()


df = pd.read_csv("../datasets/iris/iris.data")
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
sp_df = df[['sepal_length', 'petal_length']].to_numpy()

y = df['class'].to_numpy()
y = np.where(y == 'Iris-setosa', 0, y)
y = np.where(y == 'Iris-versicolor', 1, y)
y = np.where(y == 'Iris-virginica', 2, y)
y = y.astype(int)
y_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

scatter_plot(df['sepal_length'], df['petal_length'], y,
             title='Original data',
             class_names=y_names)

# -------------------
# z-scores zero-mean
# -------------------
print("-------------------")
print("z-scores zero-mean")
print("-------------------")
z_scores_zero_mean = eda_std.with_std_dev(sp_df, zero_mean=True)

scaler = StandardScaler()  # defaults to zero-mean, unit-variance
scaled_data = scaler.fit_transform(sp_df)

print('Is z_scores_zero_mean allclose to sklearn:',
      np.allclose(scaled_data, z_scores_zero_mean))
print("std: ", np.std(z_scores_zero_mean, axis=0),
      "\nmean: ", np.mean(z_scores_zero_mean, axis=0))

scatter_plot(z_scores_zero_mean[:, 0],
             z_scores_zero_mean[:, 1],
             y,
             title='z-scores with mean 0',
             class_names=y_names)

# -------------------
# z-scores NOT zero-mean
# -------------------
print("-------------------")
print("z-scores NOT zero-mean")
print("-------------------")
z_scores_not_zero_mean = eda_std.with_std_dev(sp_df, zero_mean=False)

scaler_no_mean = StandardScaler(with_mean=False)
scaled_data_no_mean = scaler_no_mean.fit_transform(sp_df)

print('Is z_scores_not_zero_mean allclose to sklearn:',
      np.allclose(scaled_data_no_mean, z_scores_not_zero_mean))
print("std: ", np.std(z_scores_not_zero_mean, axis=0),
      "\nmean: ", np.mean(z_scores_not_zero_mean, axis=0))

scatter_plot(z_scores_not_zero_mean[:, 0],
             z_scores_not_zero_mean[:, 1],
             y,
             title='z-scores with NOT mean 0',
             class_names=y_names)

# -------------------
# min-max normalization
# -------------------
print("-------------------")
print("min-max normalization")
print("-------------------")
Z_minmax = eda_std.min_max_norm(sp_df)
scaler_mm = MinMaxScaler()
scaled_data_mm = scaler_mm.fit_transform(sp_df)

print('Is min-max norm allclose to sklearn:',
      np.allclose(scaled_data_mm, Z_minmax))
print("std: ", np.std(Z_minmax, axis=0),
      "\nmean: ", np.mean(Z_minmax, axis=0))

scatter_plot(Z_minmax[:, 0],
             Z_minmax[:, 1],
             y,
             title='min-max normalization',
             class_names=y_names)

# -------------------
# Sphering
# -------------------
print("-------------------")
print("Sphering")
print("-------------------")
Z_sphere = eda_std.sphering(sp_df)

# Compare with PCA(whiten=True).
# Note that PCA whitening can differ by a rotation or sign from your direct sphering,
# so we might not get an exact match with allclose(...). We remove flipping/negating here.
pca = PCA(whiten=True)
pca_data = pca.fit_transform(sp_df)

# Check direct elementwise match
exact_match = np.allclose(pca_data, Z_sphere)
print('Is sphering allclose to sklearn:', exact_match)

print("std: ", np.std(Z_sphere, axis=0),
      "\nmean: ", np.mean(Z_sphere, axis=0))

scatter_plot(Z_sphere[:, 0],
             Z_sphere[:, 1],
             y,
             title='Sphering (pyEDAkit)',
             class_names=y_names)

scatter_plot(pca_data[:, 0],
             pca_data[:, 1],
             y,
             title='Sphering (PCA whiten)',
             class_names=y_names)


# Optional: if you want to see if they match up to an orthonormal transform:
# We'll check if one is just a rotation/reflection of the other.
# For that, we can do a Procrustes alignment or something simpler:
def rotation_tolerant_allclose(A, B, atol=1e-5):
    """
    Returns True if A and B differ only by an orthonormal transformation
    (rotation/reflection) and possibly a sign flip or row permutations.
    For a quick hack, we can try singular value decomposition:
    A ~ U S V^T, B ~ ...
    If the shapes are (n_samples, 2), we'll see if we can align them.
    """
    # Center them
    A_mean = A.mean(axis=0)
    B_mean = B.mean(axis=0)
    A_cent = A - A_mean
    B_cent = B - B_mean

    # We do SVD-based alignment
    # M = B_cent^T @ A_cent
    # If A is just a rotation of B, we can find that rotation.
    M = B_cent.T @ A_cent
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # rotation/reflection

    # Align B
    B_aligned = B_cent @ R.T

    # Compare
    return np.allclose(A_cent, B_aligned, atol=atol)


print("Rotation-tolerant match for sphering vs PCA whiten: ",
      rotation_tolerant_allclose(Z_sphere, pca_data))

plt.show()
