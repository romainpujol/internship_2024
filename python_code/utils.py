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

################################################################################


"""
    Computes the norm of the difference of two numpy arrays. Optional parameter
    p set o 2 to compute the euclidean difference.  
"""
def norm(xi_1: np.ndarray, xi_2: np.ndarray, p=2) -> float:
    return np.linalg.norm(xi_1-xi_2, p)

"""
    Given two discrete distribution of support size m and n respectively,
    compute the matrix C = (c_ij) defined by 
        c_ij = norm(xi_1-xi_2, p)^l
    where
        - xi_1 and xi_2 are the supports of the two distributions
        - optional parameter p set to 2 by default
"""
def init_distanceMatrix(xi_1: DiscreteDistribution, xi_2: DiscreteDistribution, l=2):
    atoms_1 = xi_1.get_atoms()
    atoms_2 = xi_2.get_atoms()
    
    n_i = len(atoms_1)
    n_j = len(atoms_2)
    
    d_mat = np.zeros((n_i, n_j))
    
    for i in range(n_i):
        for j in range(n_j):
            d_mat[i, j] = norm(atoms_1[i], atoms_2[j], l)
    
    return d_mat

"""
    Given an integer n, construct a DiscreteDistribution from a n sample of Normal(10,2) and a n sample of Gamma(2,2). 
"""
def generate_data_normalgamma(n:int):
    x = np.random.normal(loc=10, scale=2, size=n)
    y = np.random.gamma(shape=2, scale=2, size=n)
    
    data = np.column_stack((x, y))
    probabilities = np.full(n, 1/n)

    return DiscreteDistribution(data, probabilities)