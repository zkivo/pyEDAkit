import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

from pyEDAkit import nonlinear as eda_nonlin

def generate_s_curve(N=2000, K=12, d=2):
    # Generate true manifold
    tt = np.linspace(-np.pi, 0.5 * np.pi, int(0.6 * 10 * np.pi))
    uu = tt[::-1]
    hh = np.linspace(0, 5, 11)

    xx = np.concatenate([np.cos(tt), -np.cos(uu)])[:, None] * np.ones(len(hh))[None, :]
    yy = np.ones(len(tt) + len(uu))[:, None] * hh[None, :]
    zz = np.concatenate([np.sin(tt), 2 - np.sin(uu)])[:, None] * np.ones(len(hh))[None, :]
    cc = np.concatenate([tt, uu])[:, None] * np.ones(len(hh))[None, :]

    # Plot true manifold
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(xx, yy, zz, facecolors=plt.cm.jet(cc), rstride=1, cstride=1, edgecolor='none')
    ax1.view_init(12, -20)
    ax1.set_title("True Manifold")
    plt.show()

    # Generate sampled data
    angle = np.pi * (1.5 * np.random.rand(N // 2) - 1)
    height = 5 * np.random.rand(N)
    X = np.array([
        np.concatenate([np.cos(angle), -np.cos(angle)]),
        height,
        np.concatenate([np.sin(angle), 2 - np.sin(angle)])
    ])

    # transpose to (n_samples, n_features)
    X = X.T

    # Scatterplot of sampled data
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.concatenate([angle, angle]), cmap='jet', marker='+')
    ax2.view_init(12, -20)
    ax2.set_title("Sampled Data")
    plt.show()

    # Apply LLE
    Z = eda_nonlin.LLE(X, d, K)

    fig3 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=np.concatenate([angle, angle]), cmap='jet', marker='+')
    plt.title("LLE Embedding")
    plt.show()

    # Apply ISOMAP
    Z = eda_nonlin.ISOMAP(X, d, K)

    fig4 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=np.concatenate([angle, angle]), cmap='jet', marker='+')
    plt.title("ISOMAP Embedding")
    plt.show()

    # Apply HLLE
    Z = eda_nonlin.HLLE(X, d, K)

    fig5 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=np.concatenate([angle, angle]), cmap='jet', marker='+')
    plt.title("HLLE Embedding")
    plt.show()


def generate_swiss_hole(N=2000, K=12, d=2):
    # Generate Swiss roll manifold with sklearn including a hole
    X, t = make_swiss_roll(n_samples=N, noise=0.0, hole=True)

    # Plot Swiss roll with a hole
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='jet', marker='+')
    ax1.view_init(12, -20)
    ax1.set_title("Swiss Hole Manifold")
    plt.show()

    # Apply LLE
    Z = eda_nonlin.LLE(X, d, K)

    fig2 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=t, cmap='jet', marker='+')
    plt.title("LLE Embedding")
    plt.show()

    # Apply ISOMAP
    Z = eda_nonlin.ISOMAP(X, d, K)

    fig3 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=t, cmap='jet', marker='+')
    plt.title("ISOMAP Embedding")
    plt.show()

    # Apply HLLE
    Z = eda_nonlin.HLLE(X, d, K)

    fig4 = plt.figure(figsize=(8, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=t, cmap='jet', marker='+')
    plt.title("HLLE Embedding")
    plt.show()

generate_s_curve()
generate_swiss_hole()
