################################################################################
######################### Discrete Scenario Reduction ##########################
################################################################################

"""
    In this file we gather functions that solve the Discrete Scenario Reduction (DSR) problem.
"""

################################################################################
########################### Imports and types ##################################
################################################################################

import numpy as np

from utils import *

################################################################################
######################## Dupacova algo and variants ############################
################################################################################

"""
    Given a DiscreteDistribution xi and a desired number of output atoms m,
    compute the m indexes of the reduced distribution following the
    Forward Dupacova algorithm. Also output the value of the
    l-Wasserstein distance between xi and the reduced distribution.
"""
def dupacova_forward(xi:DiscreteDistribution, m:int, l=2):
    D = init_distanceMatrix(xi, xi, l)
    n = len(xi)
    best_d = np.inf
    index_to_chose = set(range(n))
    minimum = np.full(n, np.inf)
    reduced_indexes = [0]*m

    if len(distribution_p)==0:
        distribution_p = [1/n]*n
    distribution_p = xi.probabilities

    for k in range(m):
        for i in index_to_chose:
            minimum_i=minimum.copy()
            minimum_i=np.minimum(minimum_i, D[i])
            distance = np.dot(minimum_i, distribution_p)
            if distance<best_d:
                index=i
                best_m=minimum_i
                best_d=distance
        minimum=best_m
        reduced_indexes[k] = index
        index_to_chose.remove(index)

    return (reduced_indexes, best_d)


# xi[indices]