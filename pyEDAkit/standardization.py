from ast import Lambda
import numpy as np

def with_std_dev(data : np.ndarray, zero_mean=True):
    z_scores = None
    stds_data  = np.std(data, axis=0)
    means_data = np.mean(data, axis=0)
    if zero_mean:
        #  z_score variables will have:
        #  mean = 0
        #  variance = 1
        z_scores = (data - means_data) / stds_data
    else:
        #  z_score variables will have:
        #  mean = mean / std
        #  variance = 1
        z_scores = data / stds_data
    return z_scores

def with_range(data : np.ndarray, bounded=True):
    # Also called min-max normalization,
    # it restrains the data into the range between 1 and 0
    # With the bounded variable set as True, the data are also
    # at each end point.
    z  = None
    min = np.min(data, axis=0)
    max = np.max(data, axis=0)
    if bounded:
        z = (data - min) / (max - min)
    else:
        z = data / (max - min)
    return z

def sphering(data : np.ndarray):
    n_samples, p = data.shape
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    S = np.dot(data_centered.T, data_centered) / (n_samples - 1)
    eigenvalues, Q = np.linalg.eigh(S)
    lambda_inv_square = np.diag(1.0 / np.sqrt(eigenvalues))
    # Apply the transformation for each observation
    Z = []
    for i in range(n_samples):
        Z_i = np.dot(lambda_inv_square, np.dot(Q.T, data_centered[i,:]))
        Z.append(Z_i)
    return np.array(Z)
