from pyEDAkit import standardization as eda_std
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

sp_length = df[['sepal_length', 'petal_length']].to_numpy()
scatter(df['sepal_length'], df['petal_length'], title='Original data')

print("-------------------")
print("z-scores zero-mean")
print("-------------------")
z_scores_zero_mean = eda_std.with_std_dev(sp_length, zero_mean=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(sp_length)
print('Is z_scores_zero_mean allclose to sklearn:',
      np.allclose(scaled_data, z_scores_zero_mean))
print("std: ", np.std(z_scores_zero_mean, axis=0),
      "\nmean: ", np.mean(z_scores_zero_mean, axis=0))
scatter(z_scores_zero_mean[:, 0], z_scores_zero_mean[:, 1], title='z_scores with mean 0')

print("-------------------")
print("z-scores NOT zero-mean")
print("-------------------")
z_scores_not_zero_mean = eda_std.with_std_dev(sp_length, zero_mean=False)
scaler = StandardScaler(with_mean = False)
scaled_data = scaler.fit_transform(sp_length)
print('Is z_scores_not_zero_mean allclose to sklearn:',
      np.allclose(scaled_data, z_scores_not_zero_mean))
print("std: ", np.std(z_scores_not_zero_mean, axis=0),
      "\nmean: ", np.mean(z_scores_not_zero_mean, axis=0))
scatter(z_scores_not_zero_mean[:, 0], z_scores_not_zero_mean[:, 1], title='z_scores with NOT mean 0')

print("-------------------")
print("min-max normalization")
print("-------------------")
Z = eda_std.min_max_norm(sp_length)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(sp_length)
print('Is min-max norm allclose to sklearn:',
      np.allclose(scaled_data, Z))
print("std: ", np.std(Z, axis=0),
      "\nmean: ", np.mean(Z, axis=0))
scatter(Z[:, 0], Z[:, 1], title='min-max normalization')

print("-------------------")
print("Sphering")
print("-------------------")
Z = eda_std.sphering(sp_length)
pca = PCA(whiten=True)
pca_data = np.fliplr(pca.fit_transform(sp_length))
print('Is sphering allclose to sklearn:',
      np.allclose(pca_data, Z))
print("std: ", np.std(Z, axis=0),
      "\nmean: ", np.mean(Z, axis=0))
scatter(Z[:, 0], Z[:, 1], title='Sphering')
scatter(pca_data[:, 0], pca_data[:, 1], title='Sphering sklearn')

plt.show()
