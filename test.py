from numpy.core.fromnumeric import size
from pyEDAkit import standardization as eda_std
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

z_scores_zero = eda_std.with_std_dev(sp_length, zero_mean=True)
z_scores_not_zero = eda_std.with_std_dev(sp_length, zero_mean=False)

scatter(df['sepal_length'], df['petal_length'], title='Original data')
scatter(z_scores_zero[:, 0], z_scores_zero[:, 1], title='z_scores with mean 0')
scatter(z_scores_not_zero[:, 0], z_scores_not_zero[:, 1], title='z_scores with NOT mean 0')

print(np.std(z_scores_zero, axis=0), np.mean(z_scores_zero, axis=0))
print(np.std(z_scores_not_zero, axis=0), np.mean(z_scores_not_zero, axis=0))

Z = eda_std.sphering(sp_length)

scatter(Z[:,0], Z[:,1], title='Shering')


plt.show()
