################################################################################
############################ General comments ##################################
################################################################################

"""
    Collection of utility functions involved in csr.py and dsr.py to avoid
    redondancy.
"""
################################################################################
########################### Imports and types ##################################
################################################################################

"""
    For most linear algebra routines, we use numpy. 

    To handle discrete distributions, we use a custom class DiscreteDistribution (instead of scipy) for tailored needs.
"""

import numpy as np
from discretedistribution import DiscreteDistribution
from scipy.spatial.distance import cdist


################################################################################

def init_costMatrix(xi_1: DiscreteDistribution, xi_2: DiscreteDistribution, l: int=2):  
    """
        Given two discrete distribution of support size m and n respectively,
        compute the matrix C = (c_ij) defined by 
            c_ij = d(xi_1,xi_2)^l
        where
            - d is a distance, by default the euclidean distance
            - xi_1 and xi_2 are the supports of the two distributions
            - optional parameter l set to 2 by default
    """
    return cdist(xi_1.get_atoms(), xi_2.get_atoms(), metric='euclidean')**l

def generate_data_normalgamma(n:int):
    """
        Given an integer n, construct a DiscreteDistribution from a n sample of Normal(10,2) and a n sample of Gamma(2,2). 
    """
    x = np.random.normal(loc=10, scale=2, size=n)
    y = np.random.gamma(shape=2, scale=2, size=n)
    
    data = np.array([[x[i], y[i]] for i in range(n)])
    probabilities = np.full(n, 1/n)

    return DiscreteDistribution(data, probabilities)

def greedy_atom_selection(xi: DiscreteDistribution, ind: np.ndarray, cost_m: np.ndarray, min_cost: np.ndarray) -> int:
    """
        Compute argmin_{i in indexes} D_l(P, R u {x_i}), assuming that one knows 
            min_{i' in R} c(x_i, x_i') for every i. 
        The DiscreteDistribution P has atoms (x_i)_i and R is a subset of the atoms of P. We add x_i to R among the atoms of P in indexes that minimizes the above criterium.
    """
    min_costs = np.minimum(min_cost, cost_m[ind])
    dist =      np.dot(min_costs, xi.probabilities)
    i_tmp =     np.argmin(dist)
    return ind[i_tmp], i_tmp

# """
#     Update the minimum distances and closest indexes for the atoms that currently have the ind-th atom as a closest neighbour
# """

# def update_min_distance(min_d: np.ndarray, cost_m: np.ndarray, ind: np.ndarray, indexes: np.ndarray, indexes_closest: np.ndarray):
#     mask = (indexes_closest == ind)  # Boolean mask for the condition
#     valid_indexes = np.where(mask)[0]  
    
#     if valid_indexes.size > 0: 
#         # Subset of cost_m for valid indices and columns in indexes
#         sub_cost_m = cost_m[valid_indexes][:, indexes]  

#         # Find the minimum along the subsetted columns
#         closest_indexes = np.argmin(sub_cost_m, axis=1)  

#         # Get the min distances
#         min_distances = sub_cost_m[np.arange(len(valid_indexes)), closest_indexes]  

#         # Update the original arrays
#         indexes_closest[valid_indexes] = closest_indexes
#         min_d[valid_indexes] = min_distances