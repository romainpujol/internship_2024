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
