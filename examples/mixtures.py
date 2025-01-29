import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def genmix(num_samples, num_components, family, pie, mu, l, B = None, D= None, A = None):
    """
    Returns random samples from a Gaussian Mixture Model.
    The arguments of this function specifies the shape of the mixture
    probability density funciton. 

    A mixture model is a combination of pdf into one single pdf, which in
    our case we have a combination of Normal pdfs.

    pie, mu, l, B, D and A are lists where each element in position i,
    corresponds to the matrix or single value of the component (Normal pdf) of
    position i. Therefore, they must have same size, and the size corresponds
    to the number of components of the mixture (i.e. the number of 
    distribution used).

    """

    # check if num_components is a positive integer
    if not isinstance(num_components, int) or num_components < 0:
        raise ValueError('num_components must be a positive integer. ' \
                         'It is the number of components to consider.')

    # check if sample is a positive integer
    if not isinstance(num_samples, int) or num_samples < 0:
        raise ValueError('samples must be a positive integer. ' \
                         'It is the number of samples to generate.')

    # check if family is either 'spherical', 'diagonal' or 'general'
    family = family.lower()
    if family not in ['spherical', 'diagonal', 'general']:
        raise ValueError('family must be either "spherical", ' \
                         '"diagonal" or "general"')

    dimensions = len(mu[0])
    sigma = []
    if family == 'spherical':
        # check if pie, mu, l, lists are same size
        if not len(pie) == len(mu) == len(l):
            raise ValueError('pie, mu, l, must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(num_components):
            sigma.append(l[i] * np.eye(dimensions))
    elif family == 'diagonal':
        # check if pie, mu, l, B lists are same size
        if not len(pie) == len(mu) == len(l) == len(B):
            raise ValueError('pie, mu, l, B must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(num_components):
            sigma.append(l[i] * B[i])
    elif family == 'general':
        # check if pie, mu, l, A, D lists are same size
        if not len(pie) == len(mu) == len(l) == len(A) == len(D):
            raise ValueError('pie, mu, l, A, D must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(num_components):
            sigma.append(l[i] * D[i] @ A[i] @ D[i].T)
    else:
        raise ValueError('Error: Uknown family :c')
    
    # Initialize Gaussian Mixture Model (without fitting)
    gmm = GaussianMixture(n_components=num_components, covariance_type='full')

    # Manually set parameters
    gmm.means_ = np.array(mu)
    gmm.covariances_ = np.array(sigma)
    gmm.weights_ = np.array(pie)

    # Compute precisions (inverse of covariance matrices)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(np.array(sigma)))  # This is required

    samples, _ = gmm.sample(num_samples)

    # Convert to DataFrame for seaborn
    df = pd.DataFrame(samples, columns=[f"Dim {i+1}" for i in range(samples.shape[1])])

    # Plot scatter matrix
    sns.pairplot(df, diag_kind="hist", plot_kws={"alpha": 0.5})
    plt.show()

if __name__ == '__main__':
    
    # Spherical example
    pie = [0.7, 0.3]
    mu = [np.array([2, 2, 2]), np.array([-2, -2, -2])]
    l = [1, 2]
    genmix(250, 2, 'spherical', pie, mu, l)

    # diagonal example
    pie = [0.7, 0.3]
    mu = [np.array([2, 2]), np.array([-2, -2])]
    l = [1, 1]
    B1 = [[3, 0], 
          [0, float(1/3)]]
    B2 = [[float(1/2), 0],
          [0,          2]]
    B = [np.array(B1), np.array(B2)]
    genmix(250, 2, 'diagonal', pie, mu, l, B=B)

    # general example
    pie = [0.7, 0.3]
    mu = [np.array([2, 2]), np.array([-2, -2])]
    l = [1, 1]
    A1 = [[3, 0], 
          [0, float(1/3)]]
    A2 = [[float(1/2), 0],
          [0,          2]]
    A = [np.array(A1), np.array(A2)]
    D1 = [[ math.cos(math.pi / 8.0),       -math.sin(math.pi / 8.0)],
          [ math.sin(math.pi / 8.0),        math.cos(math.pi / 8.0)]]
    # D2 is a bit different from the example from the book because
    # they create a matrix of rank 1 and it makes some errors
    # However this example is close enough to the one of the book 
    D2 = [[ math.cos(6.0 * math.pi / 8.0),  math.sin(6.0 * math.pi / 8.0)],
          [-math.sin(6.0 * math.pi / 8.0),  math.cos(6.0  *math.pi / 8.0)]]
    D = [np.array(D1), np.array(D2)]
    genmix(250, 2, 'general', pie, mu, l, A=A, D=D)

