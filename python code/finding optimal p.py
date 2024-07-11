###########################################################
# Let's find an optimal p
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

def sum_p(m,p):
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

def dupacova_forward(distribution_x,m,l,distribution_p):
    #we assume that the distribution is uniform as it is meant to come from sampling.
    D = matrice_distance(distribution_x,distribution_x,l)
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
            minimum_i=minimum_vector(minimum_i,D[i])
            distance=sum_p(minimum_i,distribution_p)
            if distance<best_d:
                index=i
                best_m=minimum_i
                best_d=distance
        minimum=best_m
        reduced_set.append(distribution_x[index])
        index_to_chose.remove(index)

    return (reduced_set,minimum)


def local_search_ff(distribution_x, index_red, l):
    # Vérifier que m est inférieur à n
    n = len(distribution_x)
    m = len(index_red)
    index_reduced = index_red.copy()

    if m > n - 1:
        print("choose another m, remember m < n, IMPOSSIBLE")
        print("m=",m)
        print("n=",n)
        return 0

    D = matrice_distance(distribution_x, distribution_x, l)
    minimum = [float('inf')] * n
    index_closest = [-1] * n

    for i in range(n):
        for k in index_red:
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
                    if distance_ij < distance_to_reduce:
                        best_i = i
                        best_j = j
                        best_m = m_ij
                        distance_to_reduce = distance_ij
                        break  # Sortir de la boucle dès que la première amélioration est trouvée
            if best_i != -1 and best_j != -1:
                break  # Sortir de la boucle si une amélioration est trouvée

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

def local_search_ff_modified(distribution_x, index_red, l,p):

    n = len(distribution_x)
    m = len(index_red)
    index_reduced = index_red.copy()

    if m > n - 1:
        print("choose another m, remember m < n, IMPOSSIBLE")
        print("m=",m)
        print("n=",n)
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
    eps = p*distance_to_reduce

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
                    if distance_ij < distance_to_reduce - eps:
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

def local_search_bf(distribution_x,index_red,l):
    # best-fit
    n = len(distribution_x)
    m = len(index_red)
    index_reduced=index_red.copy()
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


n = 200
m = 50
distribution=generate_data(n)
t5=time.time()
g = dupacova_forward(distribution[0],m,2,distribution[1])
t6=time.time()
setz = g[0]
val = sum(g[1])/n
index = set_to_index(setz,distribution[0])

p = [(i+1)*0.0001 for i in range(150)]
values = [0]*len(p)
timer = [0]*len(p)
t1 = time.time()
a = local_search_ff(distribution[0],index,2)
t2 = time.time()

for i in range(150):
    print(i)
    t = time.time()
    values[i]=local_search_ff_modified(distribution[0],index,2,p[i])[1]
    t0 = time.time()
    timer[i]=t0-t

t3 = time.time()
c= local_search_bf(distribution[0],index,2)
t4=time.time()

plt.figure(figsize=(10, 6))
plt.plot(p,[val]*len(p),label="Dupacova")
plt.plot(p,[a[1]]*len(p), label = "First-fit")
plt.plot(p,[c[1]]*len(p),label="Best-fit")
plt.plot(p,values,label="Modified first-fit")

plt.xlabel('p')
plt.legend()
plt.title("Efficiency comparison ")
plt.show()
