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
def test_init_distanceMatrix(m,n):
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
    def norm_l(self,x,y,l):
        value=0.
        n=len(x)
        for i in range(n):
            value+=(x[i]-y[i])**2
        return value**(l/2)

    def matrice_distance(self, distribution_1,distribution_2, l):
        n_i=len(distribution_1)
        n_j=len(distribution_2)
        matrice=np.zeros((n_i,n_j))
        for i in range(n_i):
            for j in range(n_j):
                matrice[i,j]=self.norm_l(distribution_1[i],distribution_2[j],l)
        return matrice

    def minimum_vector(self, v1,v2):
        n=len(v1)
        m=len(v2)
        if n!=m:
            print("error on dimensions")
            return 0
        else:
            v3=[0]*n
            for i in range(n):
                v3[i]=min(v1[i],v2[i])
            return v3
        
    def sum_p(self, m,p):
        n1=len(m)
        n2=len(p)
        if n1!=n2:
            print("error dimension sum_p")
            return -1
        else:
            s=0
            for i in range(n1):
                s+=p[i]*m[i]
            return s
        
    def old_dupacova_forward(self, distribution_x,m,l,distribution_p):
        D = self.matrice_distance(distribution_x,distribution_x,l)
        n = len(distribution_x)
        minimum = [1000000000]*n
        reduced_set = []
        index_to_chose=[i for i in range(len(distribution_x))]
        best_d=10000000000
        if len(distribution_p)==0:
            distribution_p = [1/n]*n

        while len(reduced_set)<m:
            for i in index_to_chose:
                minimum_i=minimum.copy()
                minimum_i=self.minimum_vector(minimum_i,D[i])
                distance=self.sum_p(minimum_i,distribution_p)
                distance = np.dot(minimum_i, distribution_p)
                if distance<best_d:
                    index=i
                    best_m=minimum_i
                    best_d=distance
            minimum=best_m
            reduced_set.append(distribution_x[index])
            index_to_chose.remove(index)
        return (reduced_set,minimum)
    
    def generate_data_u(self, n):
        x = np.random.normal(loc=10, scale=2, size=n)
        y = np.random.gamma(shape=2, scale=2, size=n)
        probabilities = [1/n]*n
        data=[0]*n
        for i in range(n):
            data[i]=[x[i],y[i]]
        return (data,probabilities)
    
    def set_to_index(self, reduced,big):
        index=[]
        for i in range(len(reduced)):
            for j in range(len(big)):
                if reduced[i]==big[j]:
                    index.append(j)
        if len(set(index))!=len(reduced):
            print("error")
            return 0
        else:
            return index
    
    def test_distanceMatrices(self, n:int = 150, m:int = 20, l:int = 2):
        print("Test old distance_matrice = init_distanceMatrix")
        distribution = self.generate_data_u(n)
        distrib = DiscreteDistribution(np.array(distribution[0]), np.array(distribution[1]))

        mat_old = self.matrice_distance(distribution[0], distribution[0], l)
        mat_new = init_costMatrix(distrib, distrib, l)

        np.testing.assert_array_almost_equal(mat_old, mat_new)

    def test_oldvsnew(self, n:int = 150, m:int = 20, l:int = 2):
        np.random.seed(42)
        print("Test old vs new Forward Dupacova")

        # Old functions
        distribution = self.generate_data_u(n)
        t_old_start = time.time()
        old_sol = self.old_dupacova_forward(distribution[0],m,l,[1/n]*n)
        t_dupacova_old = time.time() - t_old_start
        print(f"    Time old Forward Dupacova: {t_dupacova_old}")
        old_indexes = np.array( self.set_to_index(old_sol[0], distribution[0]) )

        # New functions
        distrib = DiscreteDistribution(np.array(distribution[0]), np.array(distribution[1]))
        t_new_start = time.time()
        new_sol = dupacova_forward(distrib, m, l)
        t_dupacova_new = time.time() - t_new_start
        print(f"    Time new Forward Dupacova: {t_dupacova_new}")
        print(f"    Relative time ratio......: {(abs(t_dupacova_old - t_dupacova_new)/t_dupacova_new)*100}%")

        # Assert distance value are equal
        old_dist = np.power(np.dot( np.power(np.array(old_sol[1]), l), distribution[1] ), 1/l)
        new_dist = np.power(np.dot( np.power(np.array(new_sol[1]), l), distribution[1] ), 1/l)
        np.testing.assert_almost_equal(old_dist, new_dist)

        # Assert reduced indexes are equal
        np.testing.assert_array_equal(old_indexes, new_sol[0])

class TestBestFit(unittest.TestCase):
    def norm_l(self,x,y,l):
        value=0.
        n=len(x)
        for i in range(n):
            value+=(x[i]-y[i])**2
        return value**(l/2)

    def matrice_distance(self, distribution_1,distribution_2, l):
        n_i=len(distribution_1)
        n_j=len(distribution_2)
        matrice=np.zeros((n_i,n_j))
        for i in range(n_i):
            for j in range(n_j):
                matrice[i,j]=self.norm_l(distribution_1[i],distribution_2[j],l)
        return matrice
    
    def generate_data_u(self, n):
        x = np.random.normal(loc=10, scale=2, size=n)
        y = np.random.gamma(shape=2, scale=2, size=n)
        probabilities = [1/n]*n
        data=[0]*n
        for i in range(n):
            data[i]=[x[i],y[i]]
        return (data,probabilities)
    
    def old_dupacova_forward(self, distribution_x,m,l,distribution_p):
        D = self.matrice_distance(distribution_x,distribution_x,l)
        n = len(distribution_x)
        minimum = [1000000000]*n
        reduced_set = []
        index_to_chose=[i for i in range(len(distribution_x))]
        best_d=10000000000
        if len(distribution_p)==0:
            distribution_p = [1/n]*n

        while len(reduced_set)<m:
            for i in index_to_chose:
                minimum_i=minimum.copy()
                minimum_i=self.minimum_vector(minimum_i,D[i])
                distance=self.sum_p(minimum_i,distribution_p)
                distance = np.dot(minimum_i, distribution_p)
                if distance<best_d:
                    index=i
                    best_m=minimum_i
                    best_d=distance
            minimum=best_m
            reduced_set.append(distribution_x[index])
            index_to_chose.remove(index)
        return (reduced_set,minimum)

    def local_search_bf(self, distribution_x,index_reduced,l):
        # best-fit
        n = len(distribution_x)
        m = len(index_reduced)
        if m > n - 1:
            print("choose another m, remember m<n")
            return 0
        else:
            D = self.matrice_distance(distribution_x, distribution_x, l)
            minimum = [0] * n
            index_closest = [0] * n

            # Initial computation of minimum distances and closest indices
            for i in range(n):
                d = float('inf')
                for k in index_reduced:
                    dist = D[i][k]
                    if dist < d:
                        d = dist
                        index_closest[i] = k
                minimum[i] = d

            distance_to_reduce = sum(minimum) / n
            improvement = True

            while improvement:
                best_i = -1
                best_j = -1
                best_m = []

                for i in index_reduced:
                    m0 = minimum.copy()
                    index0 = index_closest.copy()

                    for j in range(n):
                        if index0[j] == i:
                            d = float('inf')
                            for k in index_reduced:
                                if k != i:
                                    dist = D[j][k]
                                    if dist < d:
                                        d = dist
                                        best_k = k
                            m0[j] = d
                            index0[j] = best_k

                    for j in range(n):
                        if j not in index_reduced:
                            m_ij = self.minimum_vector(D[j], m0)
                            distance_ij = sum(m_ij) / n
                            if distance_ij < distance_to_reduce:
                                best_i = i
                                best_j = j
                                best_m = m_ij
                                distance_to_reduce = distance_ij

                if best_i == -1 and best_j == -1:
                    # No improvement found
                    improvement = False
                else:
                    # Swap i and j in the reduced set
                    index_reduced.remove(best_i)
                    index_reduced.append(best_j)
                    minimum = best_m
                    # Update index_closest
                    for i in range(n):
                        d = float('inf')
                        for k in index_reduced:
                            dist = D[i][k]
                            if dist < d:
                                d = dist
                                index_closest[i] = k

            # Now we have the reduced set
            reduced_distribution = [distribution_x[i] for i in index_reduced]
            return (sum(minimum) / n,reduced_distribution)
        
    def minimum_vector(self, v1,v2):
        n=len(v1)
        m=len(v2)
        if n!=m:
            print("error on dimensions")
            return 0
        else:
            v3=[0]*n
            for i in range(n):
                v3[i]=min(v1[i],v2[i])
            return v3
        
    def set_to_index(self, reduced,big):
        index=[]
        for i in range(len(reduced)):
            for j in range(len(big)):
                if reduced[i]==big[j]:
                    index.append(j)
        if len(set(index))!=len(reduced):
            print("error")
            return 0
        else:
            return index
        
    def sum_p(self, m,p):
        n1=len(m)
        n2=len(p)
        if n1!=n2:
            print("error dimension sum_p")
            return -1
        else:
            s=0
            for i in range(n1):
                s+=p[i]*m[i]
            return s
    
    def test_oldvsnew(self, n:int = 150, m:int = 20, l:int = 2):
        print("Test old vs new Local Search")
        distribution = self.generate_data_u(n)
        red_distrib = self.old_dupacova_forward(distribution[0],m,l,[1/n]*n)[0]
        index_starters = self.set_to_index(red_distrib,distribution[0])

        old_bf = self.local_search_bf(distribution[0],index_starters, l)
        print(f"   Old distance value: {old_bf[0]}")

        distrib = DiscreteDistribution(distribution[0], np.array(distribution[1]))
        yo = BestFit(distrib, set(index_starters), 2)
        new_bf = yo.local_search()

################################################################################
########################## Functions from csr.py ###############################
################################################################################



################################################################################
# Direct tests if this file is compiled directly
if __name__ == "__main__":
    unittest.main()

