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

def with_sphering():
    pass
