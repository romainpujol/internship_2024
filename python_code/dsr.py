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

def dupacova_forward(xi:DiscreteDistribution, m: int, l: int=2, p: int=2) -> Tuple[list[int], float]:
    """
    Perform the Forward Dupacova algorithm to reduce the number of atoms in a discrete distribution.

    This function identifies a subset of `m` atoms from the discrete distribution `xi` that best represent the distribution in terms of minimizing the l-Wasserstein distance. The function returns the indices of the selected atoms and the corresponding distance metric.

    Parameters
    ----------
    xi : DiscreteDistribution
        The original discrete distribution from which a reduced subset of atoms is selected.
    m : int
        The number of atoms to select for the reduced distribution.
    l : int, optional
        The exponent in the l-Wasserstein distance metric (default is 2 for L2 norm).

    Returns
    -------
    Tuple[list[int], float]
        A tuple where the first element is a list of indices of the selected atoms, and the 
        second element is the calculated l-Wasserstein distance between the original distribution 
        and the reduced distribution after reallocation of the weights.

    Raises
    ------
    ValueError
        If `m` is greater than the number of atoms in `xi`.
    """
    D = init_costMatrix(xi, xi, l, p)
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
    Select the atom that minimizes the l-Wasserstein distance when added to the reduced distribution.

    This function computes the index of the atom in `ind` that, when added to the reduced 
    distribution, minimizes the l-Wasserstein distance between the original
    distribution and the reduced distribution after reallocation of the weights.

    Parameters
    ----------
    xi : DiscreteDistribution
        The original discrete distribution.
    ind : np.ndarray
        An array of indices representing the atoms to consider for addition to the reduced distribution.
    cost_m : np.ndarray
        A precomputed cost matrix containing the pairwise distances between atoms in `xi`.
    min_cost : np.ndarray
        An array storing the current minimum distances for each atom.

    Returns
    -------
    Tuple[int, int]
        A tuple containing the index of the selected atom and its position in the `ind` array.
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
    Abstract base class for implementing different variants of local search algorithms.

    The `LocalSearch` class provides a framework for local search algorithms aimed at 
    minimizing the l-Wasserstein distance between an original distribution and a reduced 
    distribution. Concrete implementations of this class must implement the `pick_ij` method, 
    which defines the strategy for selecting pairs of atoms to swap in the reduced distribution.

    Parameters
    ----------
    xi : DiscreteDistribution
        The original discrete distribution.
    initial_indexes : list[int]
        A list of initial indices for the atoms in the reduced distribution.
    l : int, optional
        The parameter for the Wasserstein metric (default is 2 for 2-Wasserstein
        distance).
    p : int, optional
        The parameter for the ground metric, typically used to define the type of norm (default is 2 for Euclidean norm).
    rho : float, optional
        A parameter that controls the minimum improvement required for an update to be accepted (default is 0.0).

    Attributes
    ----------
    l : int
        The parameter for the Wasserstein metric.
    m : int
        The number of atoms in the reduced distribution.
    n : int
        The number of atoms in the original distribution.
    rho : float
        The minimum improvement required for an update.
    xi : DiscreteDistribution
        The original discrete distribution.
    cost_m : np.ndarray
        The cost matrix containing pairwise distances between atoms.
    curr_d : float
        The current l-Wasserstein distance between the original distribution and the reduced distribution.
    ind_red : list[int]
        The indices of the atoms in the reduced distribution.
    dist_dupa : float
        The l-Wasserstein distance from the Dupacova algorithm.
    ind_to_choose : list[int]
        The indices of the atoms not included in the reduced distribution.
    """
    def __init__(self, xi:DiscreteDistribution, initial_indexes, l: int = 2, p: int =2, rho: float = 0.0):
        self.l = l
        self.m = len(initial_indexes)
        self.n = len(xi)
        if self.m > self.n: raise ValueError("m is greater than the number of atoms")
        self.rho = rho
        self.xi = xi
        self.cost_m = init_costMatrix(xi, xi, l, p)
        self.curr_d = np.inf
        self.ind_red, self.dist_dupa = self.init_R(rho, initial_indexes)

        # Complement of ind_red in {0, ..., n-1}
        self.ind_to_choose = list(np.setdiff1d(np.arange(self.n), self.ind_red))

        # Sorting indexes containers to facilitate insertion/deletion
        self.ind_to_choose.sort()
        self.ind_red.sort()

    def init_R(self, rho: float, init_ind: list[int]) -> Tuple[list[int], float]:
        """
        Initialize the reduced set of atoms and compute the initial l-Wasserstein distance.

        This method initializes the reduced set of atoms (`ind_red`) by either using 
        the given initial indices or by running the Dupacova algorithm (if `rho > 0`). 
        It also computes the initial l-Wasserstein distance between the original distribution 
        and the reduced distribution.

        Parameters
        ----------
        rho : float
            The parameter that controls the minimum improvement required for an update.
        init_ind : list[int]
            The initial indices for the atoms in the reduced distribution.

        Returns
        -------
        Tuple[list[int], float]
            A tuple containing the initialized indices and the corresponding l-Wasserstein distance.

        Raises
        ------
        ValueError
            If `rho` is negative.
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
        Determine whether a candidate update improves the l-Wasserstein distance sufficiently.

        This method checks whether the candidate update (specified by `trial_d`) improves 
        the l-Wasserstein distance between the original distribution and the reduced distribution 
        by a sufficient amount, as determined by `rho`.

        Parameters
        ----------
        trial_d : float
            The l-Wasserstein distance of the candidate update.

        Returns
        -------
        bool
            True if the candidate update improves the current distance sufficiently; otherwise, False.
        """
        return (trial_d < self.curr_d) if (self.rho <= 0) else (trial_d < self.curr_d - self.rho*self.dist_dupa)        

    @abstractmethod
    def pick_ij(self, indexes: np.ndarray) -> Tuple[int, int, float]:
        """
        Abstract method to select a pair of indices (i, j) for swapping atoms in the reduced distribution.

        This method should be implemented in concrete subclasses to define the strategy for selecting a pair of atoms to swap. The method should return the indices of the atoms to swap and the resulting l-Wasserstein distance.

        Returns
        -------
        Tuple[int, int, float]
            A tuple containing the indices of the atom to remove, the atom to add, and the resulting distance.
        """
        pass

    def swap_indexes(self, trial_i: int, trial_j: int):
        """
        Swap atoms in the reduced distribution and update the indices accordingly.

        This method updates the reduced distribution by removing the atom at index `trial_i` 
        and adding the atom at index `trial_j`. It also updates the internal index containers.

        Parameters
        ----------
        trial_i : int
            The index of the atom to remove from the reduced distribution.
        trial_j : int
            The index of the atom to add to the reduced distribution.
        """
        bisect.insort(self.ind_to_choose, trial_i)
        self.ind_red.pop(bisect.bisect_left(self.ind_red, trial_i))

        bisect.insort(self.ind_red, trial_j)
        self.ind_to_choose.pop(bisect.bisect_left(self.ind_to_choose, trial_j))

    def get_distance(self):
        """
        Get the current l-Wasserstein distance between the original and reduced distributions.

        Returns
        -------
        float
            The current l-Wasserstein distance.
        """
        return np.power(self.curr_d, 1/self.l)
    
    def get_reduced_atoms(self):
        """
        Get the atoms of the reduced distribution.

        Returns
        -------
        np.ndarray
            The atoms in the reduced distribution.
        """
        return self.xi.get_atoms()[self.ind_red]

    def local_search(self) -> None:
        """
        Perform the local search algorithm to minimize the l-Wasserstein distance.

        This method runs the local search algorithm by iteratively selecting pairs of atoms to swap (using `pick_ij`) and updating the reduced distribution until no further improvement is possible.
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
    """
    BestFit is a local search algorithm that iteratively selects the best pair (i, j)
    that minimizes the l-Wasserstein distance between the original distribution and
    the reduced distribution. The algorithm evaluates all possible swaps to identify
    the optimal pair that leads to the greatest improvement.

    Methods
    -------
    pick_ij() -> Tuple[int, int, float]
        Identifies the best pair of indices (i, j) such that the l-Wasserstein 
        distance is minimized when the i-th atom is removed and the j-th atom is added.
    
    bestfit_selection() -> Tuple[int, float]
        Evaluates all possible indices in `ind_to_choose` to find the optimal index j 
        that minimizes the distance when added to the reduced distribution.
    """
    def pick_ij(self) -> Tuple[int, int, float]:
        """
        Identifies the best pair of indices (i, j) such that the l-Wasserstein 
        distance is minimized when the i-th atom is removed from the reduced 
        distribution and the j-th atom is added (and weights are reallocated).

        The method iterates over all current atoms in the reduced distribution, 
        temporarily removing each one and calculating the best possible swap 
        with atoms not currently in the reduced distribution.

        Returns
        -------
        Tuple[int, int, float]
            A tuple containing:
            - The index of the atom to remove (i).
            - The index of the atom to add (j).
            - The resulting l-Wasserstein distance after the swap.
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
        Computes the best index j to add to the reduced distribution by evaluating 
        the minimum distance between the current atoms and all possible additions.

        The method leverages vectorization to efficiently compute the new distances 
        for all potential candidates and selects the one that minimizes the 
        l-Wasserstein distance.

        Returns
        -------
        Tuple[int, float]
            A tuple containing:
            - The index j of the atom to add to the reduced distribution.
            - The resulting distance associated with adding this atom.
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
        order to be accepted, where Q is the output of Dupacova'slgorithm.
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
    def __init__(self, xi:DiscreteDistribution, initial_indexes, l: int = 2, p: int = 2, rho: float = 0.0, shuffle: bool = False):
        super().__init__(xi, initial_indexes, l, p, rho)
        self.shuffle = shuffle

    def pick_ij(self) -> Tuple[int, int, float]:
        """
        Selects the first pair of indices (i, j) that provides an improvement 
        in the l-Wasserstein distance between the original and reduced distributions.

        If `shuffle` is True, the list of reduced indices is shuffled before 
        iterating to introduce randomness in the search.

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

import gurobipy as gp
import numpy as np

def milp(distrib: DiscreteDistribution, m: int, l: int = 2, p: int = 2):
    """
    Formulate and solve a Mixed-Integer Linear Programming (MILP) model for 
    the given discrete distribution.

    The MILP model aims to minimize the l-Wasserstein distance between the 
    original distribution and a reduced distribution with m atoms with optimal
    weights reallocation. The optimization is performed using Gurobi.

    Parameters
    ----------
    distrib : DiscreteDistribution
        The discrete distribution over which to compute the MILP formulation.
    m : int
        The number of elements to select in the reduced distribution.
    l : int, optional
        The order of the Wasserstein distance to use (default is 2).
    p : int, optional
        The norm to use (default is 2 for L2 norm).
    Returns
    -------
    float
        The optimal objective value of the MILP model, representing the minimized 
        l-Wasserstein distance.
    """
    
    n = len(distrib)
    cost_m = init_costMatrix(distrib, distrib, l, p)

    # Define the model
    model = gp.Model("model")
    model.setParam('OutputFlag', 0)
    model.setParam('Heuristics', 0)
    model.setParam('Threads', 0)  # Use all available threads

    try:
        # Define variables
        pi = model.addVars(n, n, vtype=gp.GRB.CONTINUOUS, name="pi")
        lambd = model.addVars(n, vtype=gp.GRB.BINARY, name="lambd")

        # Objective function
        model.setObjective(
            gp.quicksum(pi[i, j] * cost_m[i, j] for i in range(n) for j in range(n)) / n, 
            gp.GRB.MINIMIZE
        )

        # Constraints
        model.addConstrs((pi.sum(i, '*') == 1 for i in range(n)), name="row_sum")
        model.addConstr(lambd.sum() == m, name="reduction")
        model.addConstrs((pi[i, j] <= lambd[j] for i in range(n) for j in range(n)), name="activation")

        # Optimize the model
        model.optimize()

        # Return the objective value
        return np.power(model.objVal, 1/l)

    finally:
        # Dispose of the model to free up resources
        model.dispose()