import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_1D_helix(n: int, plot: bool = False):

    # Generate random theta values
    theta = np.random.uniform(0, 4 * np.pi, n)

    # Compute x, y, z coordinates for the helix
    x = np.cos(theta)
    y = np.sin(theta)
    z = 0.1 * theta

    # Combine into a data matrix
    X = np.column_stack((x, y, z))

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('1D Helix')
        ax.scatter(x, y, z)
        plt.show()

    return X

def generate_3D_helix(n: int, noise: float, plot: bool = False):
    """
    Generate a 3D helix with noise.

    Parameters:
    n (int): Number of points.
    noise (float): Noise level.

    Returns:
    tuple: A tuple containing the helix points (numpy array of shape (n, 3))
           and the labels (numpy array of shape (n,)).
    """
    # Generate t values
    t = np.linspace(1, n, n) / n
    t = t ** 1.0 * 2 * np.pi

    # Generate the 3D helix points
    X = np.column_stack([
        (2 + np.cos(8 * t)) * np.cos(t),
        (2 + np.cos(8 * t)) * np.sin(t),
        np.sin(8 * t)
    ]) + noise * np.random.randn(n, 3)

    # Generate labels
    labels = np.remainder(np.round(t * 1.5), 2).astype(np.uint8)

    if plot:
        # Plot the helix
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot with no colors based on labels
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], alpha=0.8)
        # Add labels and legend
        ax.set_title('3D Helix')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()

    return X, labels

def generate_scene(plot: bool = False):
    """
    Generate a 3D dataset with points sampled from:
    - The surface of a sphere
    - A cube
    - Lines attached to the sphere
    
    Returns:
        np.ndarray: A numpy array containing all 6000 points (shape: (6000, 3)).
    """
    # Sample from the surface of a sphere
    X1 = np.random.randn(1000)
    X2 = np.random.randn(1000)
    X3 = np.random.randn(1000)
    lambda_ = np.sqrt(X1**2 + X2**2 + X3**2)
    X1 /= lambda_
    X2 /= lambda_
    X3 /= lambda_
    sphere_points = np.column_stack((X1, X2, X3))
    
    # Sample from a cube
    X1 = np.random.rand(1000) + 2
    X2 = np.random.rand(1000) + 2
    X3 = np.random.rand(1000) + 2
    cube_points = np.column_stack((X1, X2, X3))
    
    # Sample from lines attached to the sphere
    # Line 1
    X1 = np.zeros(1000)
    X2 = np.zeros(1000)
    X3 = 2 * np.random.rand(1000) + 1
    line1 = np.column_stack((X1, X2, X3))
    
    # Line 2
    X1 = np.zeros(1000)
    X2 = np.zeros(1000)
    X3 = -2 * np.random.rand(1000) - 1
    line2 = np.column_stack((X1, X2, X3))
    
    # Line 3
    X1 = np.zeros(1000)
    X2 = 2 * np.random.rand(1000) + 1
    X3 = np.zeros(1000)
    line3 = np.column_stack((X1, X2, X3))
    
    # Line 4
    X1 = np.zeros(1000)
    X2 = -2 * np.random.rand(1000) - 1
    X3 = np.zeros(1000)
    line4 = np.column_stack((X1, X2, X3))
    
    # Combine all data
    data = np.vstack((sphere_points, cube_points, line1, line2, line3, line4))
    
    if plot:
        # Plot the data
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot each component with different colors
        ax.scatter(sphere_points[:, 0], sphere_points[:, 1], sphere_points[:, 2], c='r', label='Sphere', alpha=0.7)
        ax.scatter(cube_points[:, 0], cube_points[:, 1], cube_points[:, 2], c='g', label='Cube', alpha=0.7)
        ax.scatter(line1[:, 0], line1[:, 1], line1[:, 2], c='b', label='Line 1', alpha=0.7)
        ax.scatter(line2[:, 0], line2[:, 1], line2[:, 2], c='c', label='Line 2', alpha=0.7)
        ax.scatter(line3[:, 0], line3[:, 1], line3[:, 2], c='m', label='Line 3', alpha=0.7)
        ax.scatter(line4[:, 0], line4[:, 1], line4[:, 2], c='y', label='Line 4', alpha=0.7)
        
        # Set plot attributes
        ax.set_title('3D Scene with Sphere, Cube, and Lines')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)
        ax.legend()
        plt.show()
    
    return data