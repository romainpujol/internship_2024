################################################################################
######################### Discrete Scenario Reduction ##########################
################################################################################

# In this file we gather functions and classes that solve the Discrete
# Scenario Reduction (DSR) problem.

################################################################################
########################### Imports and types ##################################
################################################################################

import numpy as np
import bisect

from discretedistribution import *
from typing import Tuple
from utils import *

# The local search method is implemented as an abstract class. Local search
# variants will be concrete implementations of that abstract class.

from abc import ABC, abstractmethod

################################################################################
########################## Dupacova algorithm ##################################
################################################################################

def dupacova_forward(xi:DiscreteDistribution, m: int, l: int=2):
    """
        Given a DiscreteDistribution xi and a desired number of output atoms m,
        compute the m indexes of the reduced distribution following the Forward
        Dupacova algorithm; also give the vector (min_{i'} d(x_i, x_i')^l)_i which
        can then be used to reconstruct the value of the l-Wasserstein distance
        between input distribution and reduced distribution.
    """
    D = init_costMatrix(xi, xi, l)
    n = len(xi)
    if m > n:
        raise ValueError("m is greater than the number of atoms")
    index_to_chose = list(range(n))

    # For every atom i, save the minimal distance among the current atoms j
    minimum_d = np.full(n, np.inf) 

    # Reduced distribution is characterized by a m-subset of 1:n
    reduced_indexes = np.empty(m, dtype=int)

    for k in range(m):
        # Find the closest atom to add on a greedy Wasserstein-based criterium
        i_best, i_tmp = dupacova_selection(xi, np.array(index_to_chose), D, minimum_d)

        # Update 
        minimum_d = np.minimum(minimum_d, D[i_best])
        reduced_indexes[k] = i_best
        index_to_chose.pop(i_tmp)
    return(reduced_indexes, np.power(np.dot(minimum_d, xi.probabilities), 1/l))

def dupacova_selection(xi: DiscreteDistribution, ind: np.ndarray, cost_m: np.ndarray, min_cost: np.ndarray) -> int:
    """
        Compute argmin_{i in indexes} D_l(P, R u {x_i}), assuming that one knows 
            min_{i' in R} c(x_i, x_i') for every i. 
        The DiscreteDistribution P has atoms (x_i)_i and R is a subset of the atoms of P. We add x_i to R among the atoms of P in indexes that minimizes the above criterium.
    """
    min_costs = np.minimum(min_cost, cost_m[ind])
    dist =      np.dot(min_costs, xi.probabilities)
    i_tmp =     np.argmin(dist)
    return ind[i_tmp], i_tmp


################################################################################
############################ Local search class ################################
################################################################################

class LocalSearch(ABC):
    """
        For local search, there are many variants that can be considered. Thus,
        we first define local search in an abstract class. Then each variant
        would only need to implement the abstract function of the abstract class.
    """
    def __init__(self, xi:DiscreteDistribution, initial_indexes, l:int = 2):
        self.xi = xi
        self.cost_m = init_costMatrix(xi, xi, l)
        self.curr_d = np.inf
        self.ind_red = list(initial_indexes)
        self.ind_red.sort()
        self.l = l
        self.n = len(xi)
        self.m = len(initial_indexes)
        if self.m > self.n:
            raise ValueError("m is greater than the number of atoms")

        # Complement of ind_red in {0, ..., n-1}
        self.ind_to_choose = list(np.setdiff1d(np.arange(self.n), self.ind_red))
        self.ind_to_choose.sort()

    @abstractmethod
    def init_R(self):
        """
            Initialize the reduced set R subset of the support of xi by modifying
            in-place the internal variable ind_reduced.
        """
        pass
    
    @abstractmethod
    def improvement_condition(self, trial_d:float, best_d:float) -> bool:
        """
            Returns a bool which is True iff the new reduced distribution 
            R u {x_i} \ {x_j} is "improving enough" the l-Wass. between
            xi and the new reduced distrib. The "improving enough" part is the one
            that should be specified when implementing improvement_condition(...).
        """
        pass

    @abstractmethod
    def pick_ij(self, indexes: np.ndarray) -> Tuple[int, int, float]:
        """
            Given two atoms (known through their respective index), compute a pair
            of indexes (i,j) such that the i-th atom is removed from R and the j-th
            atom of Q is added to R. During that computation, the l-Wasserstein
            distance between xi and the best distribution that is supported on R u {x_i} \ {x_j}, which is also returned.
        
            Can have additional internal criteria.  
        
            It is in this function that the core difference between local search
            variants (best-fit, first-fit, random-fit) are expected to be expressed.
        """
        pass

    def swap_indexes(self, trial_i: int, trial_j: int):
        """
            Update the current indexes containers by swapping i with j
        """
        bisect.insort(self.ind_to_choose, trial_i)
        self.ind_red.pop(bisect.bisect_left(self.ind_red, trial_i))

        bisect.insort(self.ind_red, trial_j)
        self.ind_to_choose.pop(bisect.bisect_left(self.ind_to_choose, trial_j))

    def get_distance(self):
        return np.power(self.curr_d, 1/self.l)
    
    def get_reduced_atoms(self):
        return self.xi.get_atoms()[self.ind_red]

    def local_search(self) -> None:
        """
            Outline of the local_search algorithm, that depends on the abstract
            methods of the LocalSearch class. Namely, the abstract methods are init_R(),
            improvement_condition(...) and most importantly, pick_ij().
        """
        # If needs be, additional routine to initialize the "first guess"
        self.init_R()
        
        # Container for the best current distance, without the power 1/l
        self.curr_d = np.dot(self.xi.probabilities, 
                        np.min(self.cost_m[:, self.ind_red], axis=1))
       
        improvement = True
        while improvement:
            trial_i, trial_j, trial_d = self.pick_ij()
            if self.improvement_condition(trial_d): 
                self.curr_d = trial_d
                self.swap_indexes(trial_i, trial_j)
            else:
                improvement = False
    
################################################################################
############## Local Search concrete implementation: Best Fit ##################
################################################################################

class BestFit(LocalSearch):
    def init_R(self):
        pass # No specific initialization for BestFit

    def improvement_condition(self, trial_d: float) -> bool:
        return (trial_d < self.curr_d)

    def pick_ij(self) -> Tuple[int, int, float]:
        """
            Computes the couple (i,j) such that D_l^l(P, R \ {x_i} u {x_j}) is minimized.
        """
        # Holder for ( D_l^l(P, R \ {x_i} \cup x_J[i] )_{1 \leq i' \leq n}
        dist = np.full(self.n, np.inf)

        # Holder for the chosen J[i] among all j in ind_to_choose
        J = dict()

        for (i, ind) in enumerate(self.ind_red):
            # TODO: could use bissect to update instead of recreating list
            J[ind], dist[i] = self.bestfit_selection([k for k in self.ind_red if k!=ind])
        i_best = np.argmin(dist)
        return (self.ind_red[i_best], J[self.ind_red[i_best]], dist[i_best])
 
    def bestfit_selection(self, indexes: list[int]) -> Tuple[int, float]:
        """
        Compute J(i) = argmin_{j \in ind_to_choose} < P, v[j] > where 
            v[j] := [ min_{ z \in Ri \cup {x[j]} } c( x_k, z ) ]_{0 \leq k \leq n-1}
         and also return the associated value.

         Observe that for every 1 \leq k \leq n-1 we have
            v[j][k] = min[ w_k, c(x_k, x_j) ],
        where w_k = min_{z \in Ri} c(x_k,z). That is, we have
            v[j] = min[ w, C ],
        where w = (w_k)_k and C = (c(x_k, x_j))_k. 

        The above decomposition allows us to vectorize the computation of v[j].
        """
        # Compute the vector w (see docstring)
        min_on_Ri = np.min(self.cost_m[:, indexes], axis=1)

        # Compute v = (v[j])_{j \in ind_to_choose} uin a n x (n-m) matrix
        combined_min = np.minimum(min_on_Ri[:, np.newaxis], self.cost_m[:, self.ind_to_choose])

        # Compute (< P, v[j] >)_{j \in ind_to_choose}
        obj_val = np.dot(combined_min.transpose(), self.xi.probabilities)

        # obj_val = []
        # for j in self.ind_to_choose:
        #     combined_min = np.minimum(min_on_Ri, self.cost_m[:,j])
        #     obj_val.append(np.dot(combined_min, self.xi.probabilities))
        best_ind = np.argmin(np.array(obj_val))

        return (self.ind_to_choose[best_ind], obj_val[best_ind])
       
################################################################################
############ Local Search concrete implementation: First Fit ###################
################################################################################
    
class FirstFit(LocalSearch):

    def init_R(self):
       pass # No specific initialization for FirstFit

    def improvement_condition(self, trial_d: float) -> bool:
       return (trial_d != -1.0) 

    def pick_ij(self) -> Tuple[int, int, float]:
        dist = np.inf

        for i in self.ind_red:
            # TODO: could use bissect to update
            j, dist = self.firstfit_selection([k for k in self.ind_red if k!=i])
            if j != -1:
                print(f"i: {i}, j: {j}, curr_dist: {self.curr_d}")
                return (i, j, dist)
        return (-1, -1, -1.0)

    def firstfit_selection(self, indexes: list[int]) -> Tuple[int, float]:
        """
        Compute J(i) = argmin_{j \in ind_to_choose} < P, v[j] > where 
            v[j] := [ min_{ z \in Ri \cup {x[j]} } c( x_k, z ) ]_{0 \leq k \leq n-1}
         and also return the associated value.avec sa voix bitchy)
        """
        # Save [ min_{ z \in Ri  } c(x_k, z) ]_{0 \leq k \leq n-1} =: w
        current_min = np.min(self.cost_m[:, indexes], axis=1)

        trial_d = np.inf
        for j in self.ind_to_choose:
            # v[j] = min(w, c(x_k, x_j))
            combined_min = np.minimum(current_min, self.cost_m[:, j].flatten())

            trial_d = np.dot(self.xi.probabilities, combined_min)
            if trial_d < self.curr_d:
                print(f"trial_d = {trial_d}")
                return (j, trial_d)
        return (-1, np.inf)