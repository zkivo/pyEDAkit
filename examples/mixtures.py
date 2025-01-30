import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from itertools import combinations

def genmix(num_samples, n_components, family, pie, mu, l, B = None, D= None, A = None, plot = False):
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

    Parameters
    ----------
    num_samples : int
        The number of samples to generate.
    n_components : int
        The number of components (Normal functions) to consider.
    family : str
        The family of the covariance matrices. It can be 'spherical',
        'diagonal' or 'general'.
    pie : list
        The weights of the components. The sum of the weights must be 1.
        pie[i] is the probability that component i is selected as distribution.
    mu : list
        The means of the components. mu[i] is the mean of the component i.
    l : list
        Lambda. The scaling factors of the components. 
        l[i] is the scaling factor of the component i.
    B : list
        The diagonal matrices of the components. B[i] is the diagonal matrix
        of the component i. It is only used when family is 'diagonal'.
    D : list
        The diagonal matrices of the components. D[i] is the diagonal matrix
        of the component i. It is only used when family is 'general'.
    A : list
        The diagonal matrices of the components. A[i] is the diagonal matrix
        of the component i. It is only used when family is 'general'.
    plot : bool
        Whether to plot the samples.

    Raises
    ------
    ValueError
        If n_components is not a positive integer.
        If num_samples is not a positive integer.
        If family is not 'spherical', 'diagonal' or 'general'.
        If pie, mu, l, B, D, A lists are not same size.
    
    Returns
    -------
    numpy.ndarray 
        The generated samples (n_samples, n_features).
    sigma : list
        The covariance matrices of the components.

    """

    # check if n_components is a positive integer
    if not isinstance(n_components, int) or n_components < 0:
        raise ValueError('n_components must be a positive integer. ' \
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
    sigma = [] # convariances
    if family == 'spherical':
        # check if pie, mu, l, lists are same size
        if not len(pie) == len(mu) == len(l):
            raise ValueError('pie, mu, l, must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(n_components):
            sigma.append(l[i] * np.eye(dimensions))
    elif family == 'diagonal':
        # check if pie, mu, l, B lists are same size
        if not len(pie) == len(mu) == len(l) == len(B):
            raise ValueError('pie, mu, l, B must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(n_components):
            sigma.append(l[i] * B[i])
    elif family == 'general':
        # check if pie, mu, l, A, D lists are same size
        if not len(pie) == len(mu) == len(l) == len(A) == len(D):
            raise ValueError('pie, mu, l, A, D must have same size. ' \
                            'Each element in position i corresponds to a ' \
                            'matrix or value of the component of ' \
                            'position i.')
        for i in range(n_components):
            sigma.append(l[i] * D[i] @ A[i] @ D[i].T)
    else:
        raise ValueError('Error: Uknown family :c')
    
    # Initialize Gaussian Mixture Model (without fitting)
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')

    # Manually set parameters
    gmm.means_ = np.array(mu)
    gmm.covariances_ = np.array(sigma)
    gmm.weights_ = np.array(pie)

    # Compute precisions (inverse of covariance matrices)
    gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(np.array(sigma)))  # This is required

    samples, _ = gmm.sample(num_samples)

    # Convert to DataFrame for seaborn
    df = pd.DataFrame(samples, columns=[f"Dim {i+1}" for i in range(samples.shape[1])])

    if plot:
        sns.pairplot(df, diag_kind="hist", plot_kws={"alpha": 0.5})
        plt.show()

    return samples, sigma

def mbcfinmix(X, n_components=2, family='general', plot=False):
    """
    Estimates a Gaussian Mixture Model using the EM algorithm.
    
    Parameters:
    -----------
    X : numpy array
        The dataset (n_samples, n_features)
    n_components : int
        The number of Gaussian components.
    family : str
        The family of the covariance matrices. It can be 'spherical',
        'diagonal' or 'general'.
    plot : bool
        Whether to plot the 3D probability density function for all pairs of dimensions.

    Raises:
    -------
    ValueError
        If family is not 'spherical', 'diagonal' or 'general'.

    Returns:
    --------
    weights : numpy array
        The weights of each Gaussian component.
    means : numpy array
        The means of each Gaussian component.
    covariances : numpy array
        The covariance matrices of each Gaussian component.
    """

    # check if family is either 'spherical', 'diagonal' or 'general'
    family = family.lower()
    if family not in ['spherical', 'diagonal', 'general']:
        raise ValueError('family must be either "spherical", ' \
                         '"diagonal" or "general"')

    if family == 'spherical':
        family = 'spherical'
    elif family == 'diagonal':
        family = 'diag'
    elif family == 'general':
        family = 'full'
    else:
        raise ValueError('Error: Uknown family :c')

    gmm = GaussianMixture(n_components=n_components, covariance_type=family, random_state=42, tol=1e-6, max_iter=1000, n_init=100)
    gmm.fit(X)
    
    # Plots the 3D probability density function for all pairs of dimensions.
    if plot:
        n_features = X.shape[1]
        pairs = list(combinations(range(n_features), 2))
        for i, (dim1, dim2) in enumerate(pairs):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            x = np.linspace(X[:, dim1].min(), X[:, dim1].max(), 100)
            y = np.linspace(X[:, dim2].min(), X[:, dim2].max(), 100)
            X_grid, Y_grid = np.meshgrid(x, y)
            pos = np.dstack((X_grid, Y_grid))
            
            Z = np.zeros(X_grid.shape)
            for w, mean, cov in zip(gmm.weights_, gmm.means_, gmm.covariances_):
                mean_2d = mean[[dim1, dim2]]
                cov_2d = cov[np.ix_([dim1, dim2], [dim1, dim2])]
                rv = multivariate_normal(mean_2d, cov_2d)
                Z += w * rv.pdf(pos)
            
            ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=1)
            ax.set_xlabel(f'Feature {dim1}')
            ax.set_ylabel(f'Feature {dim2}')
            ax.set_zlabel('Density')
            plt.title(f'3D Density Plot for Features {dim1} and {dim2}')
            plt.show()

    return gmm.weights_, gmm.means_, gmm.covariances_

if __name__ == '__main__':
    # Spherical example
    pie = [0.7, 0.3]
    mu = [np.array([2, 2, 2]), np.array([-2, -2, -2])]
    l = [1, 2]
    genmix(250, 2, 'spherical', pie, mu, l, plot=True)

    # diagonal example
    pie = [0.7, 0.3]
    mu = [np.array([2, 2]), np.array([-2, -2])]
    l = [1, 1]
    B1 = [[3, 0], 
          [0, float(1/3)]]
    B2 = [[float(1/2), 0],
          [0,          2]]
    B = [np.array(B1), np.array(B2)]
    genmix(250, 2, 'diagonal', pie, mu, l, B=B, plot=True)

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
    X, sigma = genmix(250, 2, 'general', pie, mu, l, A=A, D=D, plot=True)

    # Apply EM algorithm to estimate weights, means and covariances
    weights, means, covariances = mbcfinmix(X, n_components=2, family='general', plot=True)
    print("diff weights:", weights - np.array(pie))
    print("diff means:", means - np.array(mu))
    print("diff covariances:", covariances - np.array(sigma))
