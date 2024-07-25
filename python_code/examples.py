# Small examples, served as debugging but can also help...
# These are not unit test, but could be turned into unit tests...

import numpy as np 
from discretedistribution import DiscreteDistribution

################################################################################
######################### Functions from utils.py ##############################
################################################################################

from utils import *
"""
    Print of init_distanceMatrix(...)
"""
def test_init_distanceMatrix(m,n):
    # Create large test distributions
    atoms1 = np.random.rand(m, 2)  # 1000 atoms, each with 5 features
    probabilities1 = np.random.rand(m)
    probabilities1 /= probabilities1.sum()
    xi_1 = DiscreteDistribution(atoms1, probabilities1)

    atoms2 = np.random.rand(n, 2)  # 1000 atoms, each with 5 features
    probabilities2 = np.random.rand(n)
    probabilities2 /= probabilities2.sum()
    xi_2 = DiscreteDistribution(atoms2, probabilities2)

    print(f"Print of test_init_distanceMatrix({m}, {n}) ")
    print(init_distanceMatrix(xi_1, xi_2))
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

# Direct tests if this file is compiled directly
if __name__ == "__main__":
    print("Functions from utils")
    test_init_distanceMatrix(3,2)
    test_generate_data_normalgamma(2)

