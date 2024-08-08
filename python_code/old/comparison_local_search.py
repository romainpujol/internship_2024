###########################################################
# Comparison local search: starters
###########################################################

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
import os
import gurobipy as gp
from gurobipy import GRB

os.environ["OMP_NUM_THREADS"] = "1"


random.seed(11112002)
np.random.seed(11112002)

def norm_l(x,y,l):
    value=0.
    n=len(x)
    for i in range(n):
        value+=(x[i]-y[i])**2
    return value**(l/2)

def matrice_distance(distribution_1,distribution_2,l):
    n_i=len(distribution_1)
    n_j=len(distribution_2)
    matrice=np.zeros((n_i,n_j))
    for i in range(n_i):
        for j in range(n_j):
            matrice[i,j]=norm_l(distribution_1[i],distribution_2[j],l)
    return matrice

def minimum_vector(v1,v2):
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

def generate_data(n):
    samples_normal = np.random.normal(0,2,size=n)
    samples_normal2 =np.random.normal(5,2,size=n)
    samples_gamma = np.random.gamma(0, 2, size=n)
    samples_gamma2 = np.random.gamma(1,2,size=n)
    samples_gamma3 = np.random.gamma(2,2,size=n)

    samples_uni = np.random.uniform(0,10,size=n)

    samples = []
    for i in range(n):
        samples.append([samples_normal[i],samples_normal2[i],samples_gamma[i],samples_gamma2[i],samples_gamma3[i],samples_uni[i]])

    return (samples,[1/n]*n)

def dupacova_forward(distribution_x,m,l):
    #we assume that the distribution is uniform as it is meant to come from sampling.
    D = matrice_distance(distribution_x,distribution_x,l)
    n = len(distribution_x)
    minimum = [1000000000]*n
    reduced_set = []
    index_to_chose=[i for i in range(len(distribution_x))]
    best_d=10000000000

    while len(reduced_set)<m:
        for i in index_to_chose:
            minimum_i=minimum.copy()
            minimum_i=minimum_vector(minimum_i,D[i])
            distance=sum(minimum_i)/n
            if distance<best_d:
                index=i
                best_m=minimum_i
                best_d=distance
        minimum=best_m
        reduced_set.append(distribution_x[index])
        index_to_chose.remove(index)

    return (reduced_set,sum(minimum)/n)


def local_search_bf(distribution_x,index_reduced,l):
    # best-fit
    n = len(distribution_x)
    m = len(index_reduced)
    if m > n - 1:
        print("choose another m, remember m<n")
        return 0
    else:
        D = matrice_distance(distribution_x, distribution_x, l)
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
                        m_ij = minimum_vector(D[j], m0)
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

        return (reduced_distribution, sum(minimum) / n)

def local_search_ff(distribution_x, index_reduced, l):
    n = len(distribution_x)
    m = len(index_reduced)
    if m > n - 1:
        print("choose another m, remember m < n, IMPOSSIBLE")
        print("m=",m)
        print("n=",n)
        print(distribution_x)
        return 0

    D = matrice_distance(distribution_x, distribution_x, l)
    minimum = [float('inf')] * n
    index_closest = [-1] * n

    for i in range(n):
        for k in index_reduced:
            dist = D[i][k]
            if dist < minimum[i]:
                minimum[i] = dist
                index_closest[i] = k

    distance_to_reduce = sum(minimum) / n
    improvement = True

    cpt = 0
    while improvement:
        cpt +=1
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
                    m_ij = minimum_vector(D[j], m0)
                    distance_ij = sum(m_ij) / n
                    if distance_ij < distance_to_reduce:
                        best_i = i
                        best_j = j
                        best_m = m_ij
                        distance_to_reduce = distance_ij
                        print(f"i: {best_i}, j: {best_j}, curr_dist: {distance_ij}")
                        break  # Get out of the loop
            if best_i != -1 and best_j != -1:
                break  # Get out of the loop

        if best_i == -1 and best_j == -1:
            improvement = False
        else:
            # print(f"iter: {cpt}, i={best_i}, j={best_j}")
            index_reduced.remove(best_i)
            index_reduced.append(best_j)
            minimum = best_m

            # Mise à jour des indices les plus proches
            for i in range(n):
                d = float('inf')
                for k in index_reduced:
                    dist = D[i][k]
                    if dist < d:
                        d = dist
                        index_closest[i] = k

    reduced_distribution = [distribution_x[i] for i in index_reduced]
    return reduced_distribution, sum(minimum) / n

def local_search_ff_modified(distribution_x, index_reduced, l):
    n = len(distribution_x)
    m = len(index_reduced)
    if m > n - 1:
        print("choose another m, remember m < n, IMPOSSIBLE")
        print("m=",m)
        print("n=",n)
        print(distribution_x)
        return 0

    D = matrice_distance(distribution_x, distribution_x, l)
    minimum = [float('inf')] * n
    index_closest = [-1] * n

    for i in range(n):
        for k in index_reduced:
            dist = D[i][k]
            if dist < minimum[i]:
                minimum[i] = dist
                index_closest[i] = k

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
                    m_ij = minimum_vector(D[j], m0)
                    distance_ij = sum(m_ij) / n
                    if distance_ij < 0.95*distance_to_reduce:
                        best_i = i
                        best_j = j
                        best_m = m_ij
                        distance_to_reduce = distance_ij
                        break  # Out of the loop when a suitable improvement is found
            if best_i != -1 and best_j != -1:
                break

        if best_i == -1 and best_j == -1:
            improvement = False
        else:
            index_reduced.remove(best_i)
            index_reduced.append(best_j)
            minimum = best_m

            # Mise à jour des indices les plus proches
            for i in range(n):
                d = float('inf')
                for k in index_reduced:
                    dist = D[i][k]
                    if dist < d:
                        d = dist
                        index_closest[i] = k

    reduced_distribution = [distribution_x[i] for i in index_reduced]
    return reduced_distribution, sum(minimum) / n

def set_to_index(reduced,big):
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

def dupacova_backward(distribution_x,m,l):
    D = matrice_distance(distribution_x,distribution_x,l)
    n = len(distribution_x)
    minimum = [0]*n
    reduced_set = distribution_x.copy()
    index_to_rm=[i for i in range(n)]
    index_closest=[i for i in range(n)]


    while len(reduced_set)>m:
        d_best=10000000000
        for i in index_to_rm:
            minimum_i=minimum.copy()
            index_closest_i=index_closest.copy()
            for k in range(n):
                if index_closest_i[k]==i:
                    #then we have to compute min_{i\in R}||x_k-x_i||^l
                    minimum_over=[D[k][a] for a in index_to_rm if a!=i]
                    ind=[a for a in index_to_rm if a!=i]
                    minimum_i[k]=min(minimum_over)
                    index_closest_i[k]=ind[minimum_over.index(min(minimum_over))]
                d=sum(minimum_i)/n
            if d<d_best:
                d_best=d
                to_rm=i
                m_best=minimum_i
                best_index=index_closest_i

        minimum=m_best
        index_closest=best_index
        atom_to_rm=distribution_x[to_rm]
        reduced_set.remove(atom_to_rm)
        index_to_rm.remove(to_rm)

    return (reduced_set,sum(minimum)/n)


def milp_formulation(distribution,m):
    # for L2 norm and l=2
    n = len(distribution)
    distance = matrice_distance(distribution,distribution,2)

    #define the model
    model = gp.Model("model")
    model.setParam('Heuristics', 0)
    pi = model.addVars(n,n, vtype=GRB.CONTINUOUS, name="pi")
    lambd = model.addVars(n, vtype=GRB.BINARY, name="lambd")
    model.setObjective(sum(pi[i,j]*distance[i,j] for i in range(n) for j in range(n))/n, GRB.MINIMIZE)
    model.addConstrs(sum(pi[i,j] for j in range(n))==1 for i in range(n))
    model.addConstr((sum(lambd[j] for j in range(n))==m),name="reduction")
    for j in range(n):
        model.addConstrs((pi[i,j] <= lambd[j] for i in range(n)),name="activation")
    model.optimize()
    a = model.objVal
    model.dispose()
    return a

# """
# n = 500
# distribution=generate_data(n)

# t1 = time.time()
# z=milp_formulation(distribution[0],50)
# t2=time.time()
# print("gurobi a pris ",t2-t1,"secondes et donne une valeur de ",z)
# t3=time.time()
# a = dupacova_forward(distribution[0],50,2)
# t4=time.time()
# print("dupacova a pris ",t4-t3,"secondes et donne une valeur de ",a[1])
# """

# n = 100
# deb = 10

# distribution=generate_data(n)

# mm = [i for i in range(deb,n-deb)]

# time_dup = [0]*len(mm)
# #time_bf = [0]*len(mm)
# #time_ff = [0]*len(mm)
# time_dup_back=[0]*len(mm)
# time_milp=[0]*len(mm)



# value_dup=[0]*len(mm)
# #value_bf = [0]*len(mm)
# #value_ff = [0]*len(mm)
# value_milp = [0]*len(mm)
# value_dup_back=[0]*len(mm)

# for i in range(len(mm)):
#     print(i, "/", len(mm))
#     #run of dup forward
#     t1=time.time()
#     dup = dupacova_forward(distribution[0],mm[i],2)
#     t2=time.time()
#     time_dup[i]=t2-t1
#     value_dup[i]=dup[1]

#     #run of dup backward
#     t5=time.time()
#     ff = dupacova_backward(distribution[0],mm[i],2)
#     t6=time.time()
#     time_dup_back[i]=t6-t5
#     value_dup_back[i]=ff[1]

#     #run milp
#     t3=time.time()
#     value_milp[i] = milp_formulation(distribution[0],mm[i])
#     t4=time.time()
#     time_milp[i]=t4-t3
#     value_milp[i] = milp_formulation(distribution[0],mm[i])


# plt.figure(figsize=(10, 6))
# plt.plot(mm,value_milp,label="MILP")
# plt.plot(mm,value_dup, label = "Forward Dupacova")
# plt.plot(mm,value_dup_back,label="Backward Dupacova")

# plt.xlabel('m')
# plt.ylabel('value')
# plt.legend()
# plt.title("Efficiency comparison, n=100")
# plt.show()