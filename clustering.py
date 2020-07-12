from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors

def K_Maens(X,k):
    return k_means(X=X,n_clusters=k)

import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
print(indices)