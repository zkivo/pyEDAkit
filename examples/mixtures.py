import math
import numpy as np
from pyEDAkit import mixture as eda_mix

# Spherical example
pie = [0.7, 0.3]
mu = [np.array([2, 2, 2]), np.array([-2, -2, -2])]
l = [1, 2]
eda_mix.genmix(250, 2, 'spherical', pie, mu, l, plot=True)

# diagonal example
pie = [0.7, 0.3]
mu = [np.array([2, 2]), np.array([-2, -2])]
l = [1, 1]
B1 = [[3, 0], 
        [0, float(1/3)]]
B2 = [[float(1/2), 0],
        [0,          2]]
B = [np.array(B1), np.array(B2)]
eda_mix.genmix(250, 2, 'diagonal', pie, mu, l, B=B, plot=True)

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
X, sigma = eda_mix.genmix(250, 2, 'general', pie, mu, l, A=A, D=D, plot=True)

# Apply EM algorithm to estimate weights, means and covariances
weights, means, covariances = eda_mix.mbcfinmix(X, n_components=2, family='general', plot=True)
print("diff weights:", weights - np.array(pie))
print("diff means:", means - np.array(mu))
print("diff covariances:", covariances - np.array(sigma))
