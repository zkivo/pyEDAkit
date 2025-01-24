from pyEDAkit import nonlinear as eda_nonlin
import matplotlib.pyplot as plt
import scipy.io

# Load the MATLAB data
mat_data = scipy.io.loadmat('datasets/oronsay.mat')

# Extract data and labels
data = mat_data['oronsay']
midden = mat_data['midden']
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
midden = [item for item in midden.ravel()]

Z, u_matrix = eda_nonlin.SOM(data)

# Plot the U-Matrix (distance matrix)
plt.figure(figsize=(10, 10))
plt.title("U-Matrix of the SOM", fontsize=16)
plt.imshow(u_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Distance')

# Visualize the transformed dataset
plt.figure(figsize=(8, 8))
plt.scatter(Z[:, 1], Z[:, 0], c='blue', s=50, alpha=0.7)
plt.title('Data Points Mapped to SOM Grid')
plt.xlabel('X-axis (Grid)')
plt.ylabel('Y-axis (Grid)')

plt.show()