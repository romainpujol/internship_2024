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

def test_init_distanceMatrix(m: int, n: int):
    """
        Print of init_distanceMatrix(...)
    """
    xi_1 = dummy_DiscreteDistribution(m, 2)
    xi_2 = dummy_DiscreteDistribution(n, 2)

    print(f"Print of test_init_distanceMatrix({m}, {n}) ")
    print(init_costMatrix(xi_1, xi_2))
    print("-"*5)


def test_generate_data_normalgamma(n):
    """
        Print of generate_data_normalgamma(...)
    """
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

class TestDiscreteReallocation(unittest.TestCase):
    """
        Test of discrete_reallocation(...)
    """
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


class TestForwardDupacova(unittest.TestCase):
    """
        Comparing the old and new versions of Forward Dupacova: checking that they
        give the same values and also comparing the run times
    """
    def test_oldvsnew(self, n:int = 150, m:int = 20, l:int = 2):
        from old.c_l_approximation_comparison import dupacova_forward as old_dupacova_forward
        print("Test old vs new Forward Dupacova")

        # Old functions
        distribution = generate_data_normalgamma(n)
        t_old_start = time.time()
        old_sol = old_dupacova_forward(distribution.atoms,m,l,[1/n]*n)
        t_dupacova_old = time.time() - t_old_start
        print(f"    Time old Forward Dupacova: {t_dupacova_old:.3f}s")

        # New functions
        t_new_start = time.time()
        new_sol = dupacova_forward(distribution, m, l)
        t_dupacova_new = time.time() - t_new_start
        print(f"    Time new Forward Dupacova: {t_dupacova_new:.3f}s")
        print(f"    Relative time ratio......: {(abs(t_dupacova_old - t_dupacova_new)/t_dupacova_new)*100:.2f}%")

        # Assert distance values are equal
        old_dist = np.power(np.dot( np.array(old_sol[1]), distribution.probabilities ), 1/l)
        new_dist = new_sol[1]
        np.testing.assert_almost_equal(old_dist, new_dist)

class TestBestFit(unittest.TestCase):
    
    def test_consistency(self, n: int=127, m: int = 10, l:int= 2):
        """
            Sanity check that if one starts BestFit with the warmstart obtained from a
            first run of BestFit, then local_search() does not iterate.
        """
        print("Test consistency")
        from old.c_l_approximation_comparison import local_search_bf as old_bestfit
        from old.c_l_approximation_comparison import set_to_index

        distribution = generate_data_normalgamma(n)
        index_starters = dupacova_forward(distribution, m, l)[0]
        first_bf = old_bestfit(distribution.atoms, list(index_starters), l)
        index_starters = set_to_index(first_bf[1], distribution.atoms)

        second_bf = old_bestfit(distribution.atoms, list(index_starters), l)
        np.testing.assert_equal(first_bf[0], second_bf[0])

    def test_oldvsnew(self, n:int = 149, m:int = 20, l:int = 2):
        from old.c_l_approximation_comparison import local_search_bf as old_bestfit
        print("Test old vs new Local Search: BestFit")
        distribution = generate_data_normalgamma(n)
        index_starters = dupacova_forward(distribution, m, l)[0]
        t_old_start = time.time()
        old_bf = old_bestfit(distribution.atoms, list(index_starters), l)
        t_old = time.time() - t_old_start
        print(f"   Time old BestFit..........: {t_old:.3f}s")

        t_new_start = time.time()
        bf = BestFit(distribution, index_starters, l=l)
        new_bf = bf.local_search()
        t_new = time.time() - t_new_start
        print(f"   Time new BestFit..........: {t_new:.3f}s")
        relative_time_ratio = (abs(t_old - t_new) / t_new) * 100
        print(f"   Relative time ratio.......: {relative_time_ratio:.2f}%")

        # Assert distance values are (almost) equal
        np.testing.assert_almost_equal(np.power(old_bf[0], 1/l), new_bf[0])

class TestFirstFit(unittest.TestCase):
    
    def test_oldvsnew(self, n:int = 149, m:int = 20, l:int = 2):
        np.random.seed(42069)
        from old.comparison_local_search import local_search_ff as old_firstfit
        print("Test old vs new Local Search: FirstFit")
        distribution = generate_data_normalgamma(n)
        index_starters = dupacova_forward(distribution, m, l)[0]
        t_old_start = time.time()
        old_ff = old_firstfit(distribution.atoms, list(index_starters), l)
        t_old = time.time() - t_old_start
        print(f"   Time old FirstFit.........: {t_old:.3f}s")

        t_new_start = time.time()
        ff = FirstFit(distribution, index_starters, l=l)
        new_ff = ff.local_search()
        t_new = time.time() - t_new_start
        print(f"   Time new FirstFit.........: {t_new:.3f}s")
        relative_time_ratio = (abs(t_old - t_new) / t_new) * 100
        print(f"   Relative time ratio.......: {relative_time_ratio:.2f}%")

        # Assert distance values are (almost) equal
        print(f"new dist : {new_ff[0]}")
        np.testing.assert_almost_equal(np.power(old_ff[1], 1/l), new_ff[0])

################################################################################
########################## Functions from csr.py ###############################
################################################################################



################################################################################
# Direct tests if this file is compiled directly
if __name__ == "__main__":
    unittest.main()

