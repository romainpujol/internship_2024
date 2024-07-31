################################################################################
######################### Discrete Scenario Reduction ##########################
################################################################################

"""
    In this file we gather functions and classes that solve the Discrete
    Scenario Reduction (DSR) problem.
"""

################################################################################
########################### Imports and types ##################################
################################################################################

import numpy as np

from discretedistribution import *
from typing import Tuple
from utils import *

"""
    The local search method is implemented as an abstract class. Local search
    variants will be concrete implementations of that abstract class.
"""
from abc import ABC, abstractmethod

################################################################################
########################## Dupacova algorithm ##################################
################################################################################

"""
    Given a DiscreteDistribution xi and a desired number of output atoms m,
    compute the m indexes of the reduced distribution following the Forward
    Dupacova algorithm; also give the vector (min_{i'} d(x_i, x_i')^l)_i which
    can then be used to reconstruct the value of the l-Wasserstein distance
    between input distribution and reduced distribution.
"""

def dupacova_forward(xi:DiscreteDistribution, m: int, l: int=2):
    D = init_costMatrix(xi, xi, l)
    n = len(xi)
    if m > n:
        raise ValueError("m is greater than the number of atoms")
    index_to_chose = set(range(n))

    # For every atom i, save the minimal distance among the current atoms j
    minimum_d = np.full(n, np.inf) 

    # Reduced distribution is characterized by a m-subset of 1:n
    reduced_indexes = np.empty(m, dtype=int)

    for k in range(m):
        # Find the closest atom to add on a greedy Wasserstein-based criterium
        i_best = greedy_atom_selection(xi, index_to_chose, D, minimum_d)

        # Update 
        minimum_d = np.minimum(minimum_d, D[i_best])
        reduced_indexes[k] = i_best
        index_to_chose.remove(i_best)
    return(reduced_indexes, minimum_d)


################################################################################
############################ Local search class ################################
################################################################################

"""
    For local search, there are many variants that can be considered. Thus, we
    first define local search in an abstract class. Then each variant would only
    need to implement the abstract function of the abstract class.
"""
class LocalSearch(ABC):

    def __init__(self, xi:DiscreteDistribution, initial_indexes:set[int], l:int = 2):
        self.xi = xi
        self.cost_m = init_costMatrix(xi, xi, l)
        self.ind_reduced = initial_indexes

    """
        Initialize the reduced set R subset of the support of xi by modifying
        in-place the internal variable ind_reduced.
    """
    @abstractmethod
    def init_R(self):
        pass
    
    """
        Returns a bool which is True iff the new reduced distribution 
        R u {x_i} \ {x_j} is "improving enough" the l-Wass. between
        xi and the new reduced distrib. The "improving enough" part is the one
        that should be specified when implementing improvement_condition(...).
    """
    @abstractmethod
    def improvement_condition(self, trial_d:float, best_d:float) -> bool:
        pass

    """
        Given two atoms (known through their respective index), compute a pair
        of indexes (i,j) such that the i-th atom is removed from R and the j-th
        atom of Q is added to R. During that computation, the l-Wasserstein
        distance between xi and R u {x_i} \ {x_j}, which is also returned.
        Can have additional internal criteria.  

        It is in this function that the core difference between local search
        variants (best-fit, first-fit, random-fit) are expected to be expressed.
    """
    @abstractmethod
    def pick_ij(self, indexes, min_d) -> Tuple[int, int, float]:
        pass

    """
        Called after an update of the internal variable index_reduced to update
        accordingly index_closest

        TODO: prendre en compte que seulement les indices j tq i est le plus
        proche voisin vont changer
    """
    def update_index_closest(self, ind_closest:np.ndarray) -> None:
        ind_array =  np.array(list(self.ind_reduced))    
        rows = np.arange(self.cost_m.shape[0])[:, None]
        reduced_distances = self.cost_m[rows, ind_array]
        ind_closest[:] = ind_array[np.argmin(reduced_distances, axis=1)]

    """
        Outline of the local_search algorithm, that depends on the abstract
        methods of the LocalSearch class.
    """
    def local_search(self) -> Tuple[set[int], np.ndarray]:
        self.init_R()
        n = len(self.xi)
        m = len(self.ind_reduced)
        if m > n:
            raise ValueError("m is greater than the number of atoms")
        improvement = True

        # Init min distance vector and current (best) l-Wass. distance
        index_closest = np.full(n, 0, dtype=int)
        self.update_index_closest(index_closest)
        minimum_d = self.cost_m[np.arange(n), index_closest]
        best_d = np.dot(minimum_d, self.xi.probabilities)

        while improvement:
            trial_i, trial_j, trial_d = self.pick_ij( np.copy(index_closest), np.copy(minimum_d) )
            if self.improvement_condition(trial_d, best_d): # Update
                best_d = trial_d
                self.ind_reduced.remove(trial_i)
                self.ind_reduced.add(trial_j)
                
                self.update_index_closest(index_closest)
                minimum_d = self.cost_m[np.arange(n), index_closest]
            else:
                improvement = False
        return(best_d, self.ind_reduced)

class BestFit(LocalSearch):
    def init_R(self):
        pass # No specific initialization for BestFit

    def improvement_condition(self, trial_d: float, best_d: float) -> bool:
        return (trial_d < best_d)

    def pick_ij(self, ind_closest: np.ndarray, min_d: np.ndarray) -> Tuple[int, int, float]:
        n = len(self.xi)
        ind_red = self.ind_reduced.copy()
        ind_to_choose = set(range(n)).difference(ind_red) # maybe could precompute complementaries
        dist = np.full(n, np.inf)
        J = dict()
        for i in ind_red:
            # Remove x_i from R and update ind_closest and min_d accordingly
            ind_red.remove(i)
            ind_to_choose.add(i)

            self.update_index_closest(ind_closest)
            min_d = self.cost_m[np.arange(n), ind_closest]

            # compute best J(i) among all j in 1:n \ ind_red
            J[i] = greedy_atom_selection(self.xi, ind_to_choose, self.cost_m, min_d)
            dist[i] = np.dot(min_d, self.xi.probabilities)

            # put back x_i and loop
            ind_red.add(i)
            ind_to_choose.remove(i)
    
        i_best = np.argmin(dist)
        return i_best, J[i_best], dist[i_best]