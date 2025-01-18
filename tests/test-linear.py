import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pyEDAkit import linear as eda_lin

df = pd.read_csv("datasets/iris/iris.data")
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()
y = df[['class']].to_numpy()[:,0]
y = np.where(y == 'Iris-setosa', 0, y)
y = np.where(y == 'Iris-versicolor', 1, y)
y = np.where(y == 'Iris-virginica', 2, y)
y = y.astype(int)
y_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']



# -----------------------------------
# Principal Component Analysis (PCA)
# -----------------------------------

eda_lin.PCA(X, d=2, plot=True)

# ----------------------------------
# Singular Value Decomposition (SVD)
# It provides a way to find the PCs without explicitly calculating the 
#   covariance matrix
# ----------------------------------

eda_lin.SVD(X, plot=True)

# ----------------------------------
# Non-negative Matrix Factorization (NMF)
# A way to decompose a non-negative matrix into two non-negative matrices
# It is more efficient than standard SVD. The plot shows how much
# each component contributes to the original matrix.
# ----------------------------------

eda_lin.NMF(X, d=4, plot=True)

# ----------------------------------
# Factor Analysis
# Similar to PCA, it reduces the dimensionality of the dataset.
# It creates a linear equation for each original variables to the new d
#   variables, where d < p. (p is the number of feature of the dataset)
# Also it add a small error value at each equation to make it possible the
#   relation.
# ----------------------------------

eda_lin.FA(X, d=2, plot=True)

# ----------------------------------
# Linear Discriminant Analysis (LDA)
# A dimentionality reduction method that projects all the values into a
#   single line. The goal of the projection is to find a line that
#   makes the observation as separated as possible. 
# ----------------------------------

eda_lin.LDA(X, y, plot=True)

# ----------------------------------
# Random Projection
# It projects the points into a random subspace. Interenstigly, it was 
#   demonstrated that the distances between the data points are maintained.
# ----------------------------------

eda_lin.RandProj(X, d=3, plot=True)
