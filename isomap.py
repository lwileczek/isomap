#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.spatial.distance import cdist
from sklearn.utils.graph_shortest_path import graph_shortest_path


def make_adjacency(data, dist_func="euclidean", eps=1):
   """
   Step one of ISOMAP algorithm, make Adjacency and distance matrix

   Compute the WEIGHTED adjacency matrix A from the given data points.  Points
   are considered neighbors if they are within epsilon of each other.  Distance
   between points will be calculated using SciPy's cdist which will
   compute the D matrix for us. 

   https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

   INPUT
   ------
     data - (ndarray) the dataset which should be a numpy array
     dist_func - (str) the distance metric to use. See SciPy cdist for list of
                 options
     eps - (int/float) epsilon value to define the local region. I.e. two points
                       are connected if they are within epsilon of each other.

   OUTPUT
   ------
     adj - (ndarray) the adjacency matrix
     dist - (ndarray) the distances of each point from one another
   """
   n, m = data.shape
   dist = cdist(data.T, data.T, metric=dist_func)
   adj =  np.zeros((m, m)) + np.inf
   bln = dist < eps
   adj[bln] = dist[bln]
   short = graph_shortest_path(adj)

   return short


def isomap(d, dim=2):
    """
    take an adjacency matrix and distance matrix and compute the ISOMAP
    algorithm
    
    Take the shortest path distance matrix. This follows from the algorithm in
    class, create a centering matrix and apply it to the distance matrix D. Then
    we can compute the C matrix which will be used for the eigen-decomposion
    """

    n, m = d.shape
    h = np.eye(m) - (1/m)*np.ones((m, m))
    d = d**2
    c = -1/(2*m) * h.dot(d).dot(h)
    evals, evecs = linalg.eig(c)
    print(evals[:8].real)
    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    evals = evals[:dim] 
    evecs = evecs[:, :dim]
    z = evecs.dot(np.diag(evals**(-1/2)))

    return z.real


def plot_graph(components, x, my_title="Facial Netowork Chart", filename="faces.png"):
    """
    Plot the components and overlay some images over the chart

    plotting code inspired by:
        http://benalexkeen.com/isomap-for-dimensionality-reduction-in-python/
    """

    # new stuff 
    n, m = x.shape
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111)
    ax.set_title(my_title)
    ax.set_xlabel('Component: 1')
    ax.set_ylabel('Component: 2')

    # Show 40 of the images ont the plot
    x_size = (max(components[:, 0]) - min(components[:, 0])) * 0.08
    y_size = (max(components[:, 1]) - min(components[:, 1])) * 0.08

    print("max:", np.max(components))
    print("min:", np.min(components))
    for i in range(40):
        img_num = np.random.randint(0, m)
        x0 = components[img_num, 0] - (x_size / 2.)
        y0 = components[img_num, 1] - (y_size / 2.)
        x1 = components[img_num, 0] + (x_size / 2.)
        y1 = components[img_num, 1] + (y_size / 2.)
        img = x[:, img_num].reshape(64, 64).T
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

    # Show 2D components plot
    ax.scatter(components[:, 0], components[:, 1], marker='.',alpha=0.7)
    print(len(components))

    ax.set_ylabel('Up-Down Pose')
    ax.set_xlabel('Right-Left Pose')

    plt.savefig('img/'+filename)
    return None

if __name__ == "__main__":
    import scipy.io
    mat = scipy.io.loadmat('data/isomap.mat')
    m=mat['images']
    # D = make_adjacency(m, eps=10.4, dist_func="euclidean")
    D = make_adjacency(m, eps=386, dist_func="cityblock")
    z = isomap(D)
    plot_graph(z, x=m)

