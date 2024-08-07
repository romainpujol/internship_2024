################################################################################
######################### Discrete Scenario Reduction ##########################
################################################################################

# In this file we gather functions and classes that solve the Discrete
# Scenario Reduction (DSR) problem.

################################################################################
########################### Imports and types ##################################
################################################################################

import numpy as np

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
        i_best, i_tmp = greedy_atom_selection(xi, np.array(index_to_chose), D, minimum_d)

        # Update 
        minimum_d = np.minimum(minimum_d, D[i_best])
        reduced_indexes[k] = i_best
        index_to_chose.pop(i_tmp)
    return(reduced_indexes, np.power(np.dot(minimum_d, xi.probabilities), 1/l))


################################################################################
############################ Local search class ################################
################################################################################


class LocalSearch(ABC):
    """
        For local search, there are many variants that can be considered. Thus,
        we first define local search in an abstract class. Then each variant
        would only need to implement the abstract function of the abstract class.
    """
    def __init__(self, xi:DiscreteDistribution, initial_indexes:set[int], l:int = 2):
        self.xi = xi
        self.cost_m = init_costMatrix(xi, xi, l)
        self.ind_reduced = set(initial_indexes)
        self.l = l

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

    def update_index_closest(self, ind_reduced: np.ndarray, ind_closest:np.ndarray) -> None:
        """
            Called after an update of the internal variable index_reduced to update
            accordingly index_closest

            Given a subset J=ind_reduced of {1, 2,..., n}, computes for every i in {1, ..., n}
            the "closest" element in J. Closest here is in the sense of the matrix
            cost_m where its (i,j) coefficient is the cost of moving a unit mass
            in x_i to a unit mass in x_j.

            Update in place the third argument, ind_closest.
        """
        # Making extra sure that ind_reduced is a np.ndarray
        ind_array = np.array(list(ind_reduced))
        
        # Update in-place ind_closest
        ind_closest[:] = ind_array[np.argmin(self.cost_m[:, ind_array], axis=1)]

    def local_search(self) -> Tuple[set[int], np.ndarray]:
        """
            Outline of the local_search algorithm, that depends on the abstract
            methods of the LocalSearch class.
        """
        self.init_R()
        n = len(self.xi)
        m = len(self.ind_reduced)
        if m > n:
            raise ValueError("m is greater than the number of atoms")
        improvement = True
        print(f"init ind: {self.ind_reduced}")

        # Init min cost vector and closest elements vector
        index_closest = np.full(n, 0, dtype=int)
        self.update_index_closest(self.ind_reduced, index_closest)
        print(f"new closest: {index_closest}")

        # Container for the best distance, without the power 1/l
        best_d = np.dot(self.cost_m[np.arange(n), index_closest], self.xi.probabilities)
        print(f"current best_d^l: {best_d}")

        while improvement:
            trial_i, trial_j, trial_d = self.pick_ij( np.copy(index_closest))
            print(f"trial_d: {trial_d}")
            if self.improvement_condition(trial_d, best_d): # Update
                best_d = trial_d
                print(f"current best_d^l: {best_d}")
                self.ind_reduced.remove(trial_i)
                self.ind_reduced.add(trial_j)
                
                self.update_index_closest(self.ind_reduced, index_closest)
            else:
                improvement = False
        return(np.power(best_d, 1/self.l), self.ind_reduced)
    
################################################################################
################# Local Search concrete implementation: Best Fit ###############
################################################################################

class BestFit(LocalSearch):
    def init_R(self):
        pass # No specific initialization for BestFit

    def improvement_condition(self, trial_d: float, best_d: float) -> bool:
        return (trial_d < best_d)

    def pick_ij(self, ind_closest: np.ndarray, min_d: np.ndarray) -> Tuple[int, int, float]:
        n = len(self.xi)

        # Copy of current indexes characterizing current reduced distrib Q
        ind_red = self.ind_reduced.copy()

        # Complement of ind_red in {0, ..., n-1}
        ind_to_choose = list(np.setdiff1d(np.arange(n), np.array(list(ind_red))))

        # Container for vector of costs of moving x_i to closest in ind_to_choose
        costs_closest = np.full(n, 0.0, dtype=float)

        # Holder for ( Distance(P, R \ {x_i} \cup x_J[i] )_i
        dist = np.full(n, np.inf)

        # Holder for "the best" J[i] among all j in ind_to_choose
        J = dict()

        for i in ind_red:
            # Remove x_i from R and update ind_to_choose, ind_closest and min_d accordingly
            ind_red.remove(i)
            ind_to_choose.append(i)

            self.update_index_closest(ind_red, ind_closest)
            print(f"{i} ind_closest = {ind_closest}")
            costs_closest = np.minimum(self.cost_m[np.arange(n), ind_closest], ) 

            # j_tmp is the corresponding index of J[i] in ind_to_choose
            J[i], j_tmp = greedy_atom_selection(self.xi, np.array(ind_to_choose), self.cost_m, min_d)
            print(f"J[{i}] = {J[i]}")
            dist[i] = np.dot(min_d, self.xi.probabilities)

            # put back x_i and loop
            ind_red.add(i)
            ind_to_choose.pop()
            print(f"dist {i}: {dist[i]}")
        i_best = np.argmin(dist)
        return i_best, J[i_best], dist[i_best]
    
################################################################################
################ Local Search concrete implementation: First Fit ###############
################################################################################
    
class FirstFit(LocalSearch):
   def init_R(self):
       pass # No specific initialization for FirstFit

   def improvement_condition(self, trial_d: float, best_d: float) -> bool:
       return ( (trial_d != -1) and (trial_d < best_d))

   def pick_ij(self, ind_closest:np.ndarray, min_d:np.ndarray) -> Tuple[int, int, float]:
       n = len(self.xi)
       ind_red = self.ind_reduced.copy()
       dist = np.full(n, np.inf)

       # shuffle possible après la copie
       i_best = -1
       j_best = -1
       dist_best = -1 #trial_d = -1 si pas de changement
       to_beat = np.dot(min_d,self.xi.probabilities) #the distance to beat
       print("The distance to beat is ", to_beat)

       for i in ind_red:
           # Remove x_i from R and update ind_closest and min_d accordingly
           ind_red.remove(i)
           self.update_index_closest(self.ind_reduced, ind_closest)
           min_d = self.cost_m[np.arange(n), ind_closest]

           # compute d_ij, the distance when swapping i and j 
           for j in range(n):
               if j not in ind_red:
                   m_ij = np.minimum(min_d,self.cost_m[j]) # vecteur m avec swap ij
                   d_ij = np.dot(m_ij,self.xi.probabilities)
                   print(d_ij)

                   if d_ij < to_beat :
                       i_best = i
                       j_best = j
                       dist_best = d_ij
                       ind_red.add(i) # as we break we have to add back the atom before the break
                       break
                   
           # put back x_i and loop
           ind_red.add(i)
       return (i_best,j_best,dist_best)