import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Load the MATLAB data
mat_data = scipy.io.loadmat('datasets/leukemia.mat')

# Extract data and labels
data = mat_data['leukemia'].T  # Feature matrix
btcell = mat_data['btcell']  # Labels for T, B cells or NA
cancertype = mat_data['cancertype']  # Labels for ALL or AML

# Flatten and extract strings from btcell
btcell = [item[0] for item in btcell.ravel()]
cancertype = [item[0] for item in cancertype.ravel()]

# Ensure labels match the number of samples
if len(btcell) != data.shape[0] or len(cancertype) != data.shape[0]:
    raise ValueError("Mismatch in the number of samples between data and labels.")

# Perform metric MDS in two dimensions
mds_metric = MDS(n_components=2, random_state=42, metric=True)
data_mds_metric = mds_metric.fit_transform(data)

# Perform non-metric MDS in two dimensions
mds_nonmetric = MDS(n_components=2, random_state=42, metric=False)
data_mds_nonmetric = mds_nonmetric.fit_transform(data)

# Create DataFrames for visualization
df_metric = pd.DataFrame(data_mds_metric, columns=['Dim1', 'Dim2'])
df_metric['btcell'] = btcell
df_metric['cancertype'] = cancertype

df_nonmetric = pd.DataFrame(data_mds_nonmetric, columns=['Dim1', 'Dim2'])
df_nonmetric['btcell'] = btcell
df_nonmetric['cancertype'] = cancertype

# Plot 1: T, B cells, or NA
plt.figure(figsize=(8, 6))
for label in np.unique(df_metric['btcell']):
    subset = df_metric[df_metric['btcell'] == label]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=label, alpha=0.7)
plt.title('MDS (Metric) Plot by T, B cells or NA')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()

# Plot 2: ALL or AML
plt.figure(figsize=(8, 6))
for label in np.unique(df_metric['cancertype']):
    subset = df_metric[df_metric['cancertype'] == label]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=label, alpha=0.7)
plt.title('MDS (Metric) Plot by ALL or AML')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()