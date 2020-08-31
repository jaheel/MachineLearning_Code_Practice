import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.manifold import MDS

def calculate_dist(X):
    """
    @ param X: matrix
    @ return dist: The square of the distance between any two points
    """
    
    sum_x = np.sum(np.square(X), 1)
    dist = np.add(np.add( -2 * np.dot(X, X.T), sum_x).T, sum_x)

    return dist


def Personal_MDS(data, n_dims):
    """

    @ param data: (n_samples, n_features)
    @ param n_dims: target n_dims

    @ return Z: (n_samples, n_dims)

    """
    
    n,d = data.shape
    dist = calculate_dist(data)
    dist[dist < 0] = 0

    dist_ij = np.ones((n,n)) * np.sum(dist)/n**2
    dist_i = np.sum(dist, axis=1, keepdims=True)/n
    dist_j = np.sum(dist, axis=0, keepdims=True)/n

    B = -(dist_ij - dist_i -dist_j + dist)/2

    eig_value, eig_vector = np.linalg.eig(B)
    
    index_list = np.argsort(-eig_value)[:n_dims]
    
    picked_eig_value = eig_value[index_list].real
    picked_eig_vector = eig_vector[:, index_list]

    Z = picked_eig_vector*picked_eig_value**(0.5)
    return Z


if __name__=='__main__':
    iris = load_iris()
    data = iris.data
    Y = iris.target

    data_1 = Personal_MDS(data, 2)
    data_2 = MDS(n_components=2).fit_transform(data)

    plt.figure(figsize=(8, 4))
    plt.subplot(121)
    plt.title("Personal_MDS")
    plt.scatter(data_1[:, 0], data_1[:, 1], c=Y)

    plt.subplot(122)
    plt.title("sklearn_MDS")
    plt.scatter(data_2[:, 0], data_2[:, 1], c=Y)

    plt.show()
    

