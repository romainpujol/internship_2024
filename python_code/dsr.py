################################################################################
######################### Discrete Scenario Reduction ##########################
################################################################################

# In this file we gather functions and classes that solve the Discrete
# Scenario Reduction (DSR) problem.
#
# Currently implemented methods are:
#
# 1) (Forward) Dupacova's algorithm.
#
# 2) Local Search algorithm Rujeerapaiboon, Schindler, Kuhn, Wiesemann (2023):
# 2.1) Local Search: BestFit,
# 2.2) Local Search: FirstFit with possibly flexible decrease and/or random
# shuffling. See FirstFit for more details.
#
# 3) MILP equivalent formulation, solved using Gurobi.

################################################################################
########################### Imports and types ##################################
################################################################################

import numpy as np
import bisect
import random

# MILP equivalent formulation will be solved by Gurobi
import gurobipy as gp

from discretedistribution import *
from typing import Tuple
from utils import *

# The local search method is implemented as an abstract class. Local search
# variants will be concrete implementations of that abstract class.

from abc import ABC, abstractmethod

################################################################################
########################## Dupacova algorithm ##################################
################################################################################

def dupacova_forward(xi:DiscreteDistribution, m: int, l: int=2) -> Tuple[list[int], float]:
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
    return(list(reduced_indexes), np.power(np.dot(minimum_d, xi.probabilities), 1/l))

def dupacova_selection(xi: DiscreteDistribution, ind: np.ndarray, cost_m: np.ndarray, min_cost: np.ndarray) -> int:
    """
    Compute argmin_{i in indexes} D_l(P, R u {x_i}), assuming that one knows 
        min_{i' in R} c(x_i, x_i') for every i. 
    The DiscreteDistribution P has atoms (x_i)_i and R is a subset of the atoms of P. We add x_i to R among the atoms of in indexes that minimizes the above criterium.
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
    would only need to implement the abstract function of the abstract
    class.
    
    Optional parameter p that changes the improvement condition to only
    accept improvements of at least p*distance(P, Q) where Q is the
    result of Dupacova's algorithm.
    """
    def __init__(self, xi:DiscreteDistribution, initial_indexes, l:int = 2, rho: float = 0.0):
        self.l = l
        self.m = len(initial_indexes)
        self.n = len(xi)
        if self.m > self.n: raise ValueError("m is greater than the number of atoms")
        self.rho = rho
        self.xi = xi
        self.cost_m = init_costMatrix(xi, xi, l)
        self.curr_d = np.inf
        self.ind_red, self.dist_dupa = self.init_R(rho, initial_indexes)

        # Complement of ind_red in {0, ..., n-1}
        self.ind_to_choose = list(np.setdiff1d(np.arange(self.n), self.ind_red))

        # Sorting indexes containers to facilitate insertion/deletion
        self.ind_to_choose.sort()
        self.ind_red.sort()

    def init_R(self, rho: float, init_ind: list[int]) -> Tuple[list[int], float]:
        """
        Initialize the reduced set R subset of the support of xi by modifying
        in-place the internal variable ind_reduced. Also saves the value of
        d(P, Q) where P is input distribution and Q is the reduced one
        obtained by Forward Dupacova's algorithm.
        """
        if rho < 0:
            raise ValueError("rho should be nonnegative: {rho} was given")
        elif rho > 0:
           # Run Dupacova's algorithm to both initialize ind_red and dist_dupa
            return dupacova_forward(self.xi, self.m, self.l)
        else:
            return (list(init_ind), np.inf)

    def improvement_condition(self, trial_d:float) -> bool:
        """
        Returns a bool which is True iff the new reduced distribution 
        R u {x_i} \ {x_j} is "improving (enough)" the l-Wass. between
        xi and the new reduced distrib. 
        """
        return (trial_d < self.curr_d) if (self.rho <= 0) else (trial_d < self.curr_d - self.rho*self.dist_dupa)        

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
            Outline of the local_search algorithm that still depends on an
            implementation of pick_ij()
        """
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

    def pick_ij(self) -> Tuple[int, int, float]:
        """
            Computes the couple (i,j) such that D_l^l(P, R \ {x_i} u {x_j}) is minimized.
        """
        # Holder for ( D_l^l(P, R \ {x_i} \cup x_J[i] )_{1 \leq i' \leq n}
        dist = np.full(self.n, np.inf)

        # Holder for the chosen J[i] among all j in ind_to_choose
        J = dict()

        for (i, ind) in enumerate(self.ind_red):
            # Temporarily remove ind from ind_red, added back at the end
            self.ind_red.pop(bisect.bisect_left(self.ind_red, ind))
            J[ind], dist[i] = self.bestfit_selection()
            bisect.insort(self.ind_red, ind)
        i_best = np.argmin(dist)
        return (self.ind_red[i_best], J[self.ind_red[i_best]], dist[i_best])
 
    def bestfit_selection(self) -> Tuple[int, float]:
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
        min_on_Ri = np.min(self.cost_m[:, self.ind_red], axis=1)

        # Compute v = (v[j])_{j \in ind_to_choose} uin a n x (n-m) matrix
        combined_min = np.minimum(min_on_Ri[:, np.newaxis], self.cost_m[:, self.ind_to_choose])

        # Compute (< P, v[j] >)_{j \in ind_to_choose}
        obj_val = np.dot(combined_min.transpose(), self.xi.probabilities)
        best_ind = np.argmin(np.array(obj_val))

        return (self.ind_to_choose[best_ind], obj_val[best_ind])
       
################################################################################
############ Local Search concrete implementation: First Fit ###################
################################################################################
    
class FirstFit(LocalSearch):
    """
    FirstFit is a local search algorithm that iteratively selects the best pair (i, j)
    according to a first-fit heuristic. It attempts to improve a solution by choosing 
    the first pair that improves the current objective value.

    Parameters
    ----------
    xi : DiscreteDistribution
        The discrete distribution over which the search is performed.
    initial_indexes : list[int]
        The initial indices to be considered in the search.
    l : int, optional
        The norm used for distance calculations (default is 2 for L2 norm).
    rho : float, optional
        Proportion (default is 0.0) of d(xi, Q) that an improvement should do in
        order to be accepted, where Q is the output of Dupacova'slgorithm .
    shuffle : bool, optional
        If True, the list of indices is shuffled at each iteration to introduce randomness (default is False).

    Methods
    -------
    pick_ij() -> Tuple[int, int, float]
        Selects the best pair of indices (i, j) based on the first-fit heuristic and 
        returns them along with the associated distance.
    
    firstfit_selection(indexes: list[int]) -> Tuple[int, float]
        Computes the dot product < P, v[j] > where v[j] is the minimum distance vector 
        for a given index j, and stops as soon as an improvement is found.
    """
    def __init__(self, xi:DiscreteDistribution, initial_indexes, l:int = 2, rho: float = 0.0, shuffle: bool = False):
        super().__init__(xi, initial_indexes, l, rho)
        self.shuffle = shuffle

    def pick_ij(self) -> Tuple[int, int, float]:
        """
        Selects the best pair of indices (i, j) based on the first-fit heuristic.

        Returns
        -------
        Tuple[int, int, float]
            A tuple containing the selected pair (i, j) and the associated distance.
            If no improvement is found, returns (-1, -1, np.inf).
        """
        dist = np.inf

        # If shuffling, update in place ind_red and sort it back before returning
        if self.shuffle:
            random.shuffle(self.ind_red)

        for i in self.ind_red:
            j, dist = self.firstfit_selection([k for k in self.ind_red if k!=i])
            if j != -1:
                return (i, j, dist) if not bool else (self.ind_red.sort() or (i, j, dist))
        return (-1, -1, np.inf) if not bool else (self.ind_red.sort() or (-1, -1, np.inf))
    
    def firstfit_selection(self, indexes: list[int]) -> Tuple[int, float]:
        """
        Computes the optimal index j by evaluating the minimum cost function 
        and stops as soon as an improvement over the current distance is found.

        Parameters
        ----------
        indexes : list[int]
            A list of indices to consider for selection.

        Returns
        -------
        Tuple[int, float]
            A tuple containing the selected index j and the corresponding distance.
            If no improvement is found, returns (-1, np.inf).
        """
        min_on_Ri = np.min(self.cost_m[:, indexes], axis=1)

        trial_d = np.inf
        for j in self.ind_to_choose:
            combined_min = np.minimum(min_on_Ri, self.cost_m[:,j])
            trial_d = np.dot(self.xi.probabilities, combined_min)
            if self.improvement_condition(trial_d):
                return (j, trial_d)
        return (-1, np.inf)

##################################################################################
############################ MILP Reformulation ##################################
##################################################################################

def milp(distrib: DiscreteDistribution, m: int, l: int = 2):
    """
    Formulate and solve a MILP model for the given distribution.

    Parameters:
    distrib: DiscreteDistribution
        The distribution over which to compute the MILP formulation.
    m: int
        The number of elements to select.
    l: int
        The norm to use (default is L2 norm, l=2).

    Returns:
    float
        The optimal objective value of the MILP model.
    """
    
    n = len(distrib)
    distance = init_costMatrix(distrib, distrib, l)

    # Define the model
    model = gp.Model("model")
    # model.setParam('Heuristics', 0)
    model.setParam('OutputFlag', 0)

    try:
        # Define variables
        pi = model.addVars(n, n, vtype=gp.GRB.CONTINUOUS, name="pi")
        lambd = model.addVars(n, vtype=gp.GRB.BINARY, name="lambd")

        # Objective function
        model.setObjective(
            gp.quicksum(pi[i, j] * distance[i, j] for i in range(n) for j in range(n)) / n, 
            gp.GRB.MINIMIZE
        )

        # Constraints
        model.addConstrs((gp.quicksum(pi[i, j] for j in range(n)) == 1 for i in range(n)), name="row_sum")
        model.addConstr(gp.quicksum(lambd[j] for j in range(n)) == m, name="reduction")
        model.addConstrs((pi[i, j] <= lambd[j] for i in range(n) for j in range(n)), name="activation")

        # Optimize the model
        model.optimize()

        # Return the objective value
        return model.objVal

    finally:
        # Dispose of the model to free up resources
        model.dispose()