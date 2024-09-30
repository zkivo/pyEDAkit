import numpy as np

def with_std_dev(data : np.ndarray, zero_mean=True):
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

def with_range():
    pass

def with_sphering():
    pass
