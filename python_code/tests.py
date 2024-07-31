# A few small examples, served as debugging but can also help...
# Some are not unit test, but could be turned into unit tests...

import numpy as np 
import unittest
import time

from discretedistribution import *
from dsr import *

################################################################################
######################### Functions from utils.py ##############################
################################################################################

from utils import *
"""
    Print of init_distanceMatrix(...)
"""
def test_init_distanceMatrix(m: int, n: int):
    xi_1 = dummy_DiscreteDistribution(m, 2)
    xi_2 = dummy_DiscreteDistribution(n, 2)

    print(f"Print of test_init_distanceMatrix({m}, {n}) ")
    print(init_costMatrix(xi_1, xi_2))
    print("-"*5)

"""
    Print of generate_data_normalgamma(...)
"""
def test_generate_data_normalgamma(n):
    distribution = generate_data_normalgamma(n)
    print(f"Print of test_generate_data_normalgamma({n}) ")
    print("Distribution:")
    print(distribution)
    print("Atoms:\n", distribution.get_atoms())
    print("Probabilities:\n", distribution.get_probabilities())
    print("-"*5)


################################################################################
################ Functions from discretedistribution.py ########################
################################################################################

"""
    Test of discrete_reallocation(...)
"""
class TestDiscreteReallocation(unittest.TestCase):
    def test_discrete_reallocation(self, n:int = 10, dim:int = 4):
        print("Test discrete_reallocation")
        xi = dummy_DiscreteDistribution(n, dim)
        xi_atoms = xi.get_atoms()

        # Define indexes and new weights for reallocation
        indexes = [1, 3, 4]
        weights = np.array([0.3, 0.4, 0.3])

        # Call the discrete_reallocation function
        reallocated_dist = discrete_reallocation(xi, indexes, weights)

        # Expected atoms and probabilities after reallocation
        expected_atoms = xi_atoms[indexes]
        expected_probabilities = np.array([0.3, 0.4, 0.3])

        # Assert the atoms and probabilities are as expected
        np.testing.assert_array_equal(reallocated_dist.get_atoms(), expected_atoms)
        np.testing.assert_array_almost_equal(reallocated_dist.get_probabilities(), expected_probabilities)

################################################################################
########################## Functions from dsr.py ###############################
################################################################################

"""
    Comparing the old and new versions of Forward Dupacova: checking that they
    give the same values and also comparing the run times
"""
class TestForwardDupacova(unittest.TestCase):

    def test_oldvsnew(self, n:int = 150, m:int = 20, l:int = 2):
        from c_l_approximation_comparison import dupacova_forward as old_dupacova_forward
        from c_l_approximation_comparison import set_to_index
        np.random.seed(42069)
        print("Test old vs new Forward Dupacova")

        # Old functions
        distribution = generate_data_normalgamma(n)
        t_old_start = time.time()
        old_sol = old_dupacova_forward(distribution.atoms,m,l,[1/n]*n)
        t_dupacova_old = time.time() - t_old_start
        print(f"    Time old Forward Dupacova: {t_dupacova_old}")
        old_indexes = np.array( set_to_index(old_sol[0], distribution.atoms) )

        # New functions
        t_new_start = time.time()
        new_sol = dupacova_forward(distribution, m, l)
        t_dupacova_new = time.time() - t_new_start
        print(f"    Time new Forward Dupacova: {t_dupacova_new}")
        print(f"    Relative time ratio......: {(abs(t_dupacova_old - t_dupacova_new)/t_dupacova_new)*100}%")

        # Assert distance values are equal
        old_dist = np.power(np.dot( np.power(np.array(old_sol[1]), l), distribution.probabilities ), 1/l)
        new_dist = np.power(np.dot( np.power(np.array(new_sol[1]), l), distribution.probabilities ), 1/l)
        np.testing.assert_almost_equal(old_dist, new_dist)

        # Assert reduced indexes are equal
        np.testing.assert_array_equal(old_indexes, new_sol[0])

class TestBestFit(unittest.TestCase):
    
    def test_oldvsnew(self, n:int = 150, m:int = 20, l:int = 2):
        from c_l_approximation_comparison import dupacova_forward as old_dupacova_forward
        from c_l_approximation_comparison import local_search_bf
        from c_l_approximation_comparison import set_to_index
        np.random.seed(42069)

        print("Test old vs new Local Search")
        distribution = generate_data_normalgamma(n)
        red_distrib = old_dupacova_forward(distribution.atoms,m,l,[1/n]*n)[0]
        index_starters = set_to_index(red_distrib, distribution.atoms)
        t_old_start = time.time()
        old_bf = local_search_bf(distribution.atoms,index_starters, l)
        t_old = time.time() - t_old_start
        print(f"   Old time..........: {t_old}")

        t_new_start = time.time()
        yo = BestFit(distribution, set(index_starters), 2)
        new_bf = yo.local_search()
        t_new = time.time() - t_new_start
        print(f"   New time..........: {t_new}")
        print(f"   Rel. time ratio...: {(abs(t_old - t_new)/t_new)*100}%")

        # Assert distance values are equel
        np.testing.assert_almost_equal(old_bf[0], new_bf[0])

################################################################################
########################## Functions from csr.py ###############################
################################################################################



################################################################################
# Direct tests if this file is compiled directly
if __name__ == "__main__":
    unittest.main()

