from pyEDAkit import standardization as eda_std
from sklearn.preprocessing import StandardScaler
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

z_scores_zero     = eda_std.with_std_dev(sp_length, zero_mean=True)
z_scores_not_zero = eda_std.with_std_dev(sp_length, zero_mean=False)
z_norm_bounded    = eda_std.with_range(sp_length, bounded=True)
z_norm_not_bouded = eda_std.with_range(sp_length, bounded=False)
Z = eda_std.sphering(sp_length)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(sp_length)

print('first one:', np.allclose(scaled_data, z_scores_zero))

scatter(df['sepal_length'], df['petal_length'], title='Original data')
scatter(z_scores_zero[:, 0], z_scores_zero[:, 1], title='z_scores with mean 0')
scatter(z_scores_not_zero[:, 0], z_scores_not_zero[:, 1], title='z_scores with NOT mean 0')
scatter(z_norm_bounded[:, 0], z_norm_bounded[:, 1], title='z_norm_bounded')
scatter(z_norm_not_bouded[:, 0], z_norm_not_bouded[:, 1], title='z_norm_not_bouded')
scatter(Z[:,0], Z[:,1], title='Shering')

print(np.std(z_scores_zero, axis=0), np.mean(z_scores_zero, axis=0))
print(np.std(z_scores_not_zero, axis=0), np.mean(z_scores_not_zero, axis=0))
print(np.std(Z, axis=0), np.mean(Z, axis=0))

plt.show()
