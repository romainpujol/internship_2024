################################################################################
######################### Continuous Scenario Reduction ########################
################################################################################

# In this file we gather functions that solve the Continuous Scenario Reduction
# (CSR) problem.

################################################################################

import numpy as np

from discretedistribution import *
from utils import *
from sklearn.cluster import KMeans
from utils import init_costMatrix

################################################################################

def k_means(distribution: DiscreteDistribution, m: int, warmcentroids: np.ndarray = np.array([]), l: int=2):
    if warmcentroids.size:
        km = KMeans(init=warmcentroids, n_init=1, n_clusters=m)
        labels  = km.fit_predict(distribution.atoms, sample_weight=distribution.probabilities)
    else: # "Greedy-KMeans++ start"
        km = KMeans(n_clusters=m)
        labels  = km.fit_predict(distribution.atoms, sample_weight=distribution.probabilities)
    
    centroids = km.cluster_centers_

    # Calculate the probability weights of the centroids
    n_clusters = km.n_clusters
    probability_weights = np.zeros(n_clusters)
    initial_weights = distribution.probabilities

    for i in range(n_clusters):
        probability_weights[i] = np.sum([initial_weights[j] for j in range(len(distribution)) if (labels[j] == i)]) 

    matrice = init_costMatrix(distribution, DiscreteDistribution(centroids, probability_weights), 2)
    min_rows = np.min(matrice, axis=1)

    return np.power(np.dot(min_rows, distribution.probabilities), 1/l)

