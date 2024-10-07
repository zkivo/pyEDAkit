from pyEDAkit import standardization as eda_std
from pyEDAkit import linear as eda_lin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def scatter(x, y, title='Title'):
    plt.figure()
    color = np.arange(len(x))
    color.fill(np.random.rand())
    plt.scatter(x, y, c=color, label='data point', s=3)
    plt.axhline(0, color='gray', linestyle='--')  # Horizontal dotted line at y=0
    plt.axvline(0, color='gray', linestyle='--')  # Vertical dotted line at x=0
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.draw()

df = pd.read_csv("data/iris/iris.data")
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

Z_1 = eda_lin.PCA(X, n_components=2, covariance=True)
sk_pca = PCA(n_components=2).fit_transform(X)
sk_pca[:, 1] = -sk_pca[:, 1]
print('Is Z_1 allclose to sklearn:', np.allclose(Z_1, sk_pca))
scatter(Z_1[:, 0], Z_1[:, 1], title='my PCA with covariance')
scatter(sk_pca[:, 0], sk_pca[:, 1], title='sklearn PCA with covariance')

X_std = eda_std.with_std_dev(X, zero_mean=True)
Z_2 = eda_lin.PCA(X, n_components=2, covariance=False)
sk_pca = PCA(n_components=2).fit_transform(X_std)
sk_pca[:, 1] = -sk_pca[:, 1]
print('Is Z_2 allclose to sklearn:', np.allclose(Z_2, sk_pca))
scatter(Z_2[:, 0], Z_2[:, 1], title='my PCA with correlation')
scatter(sk_pca[:, 0], sk_pca[:, 1], title='sklearn PCA with correlation')

plt.show()