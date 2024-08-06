###########################################################
# Random first-fit
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
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

os.environ["OMP_NUM_THREADS"] = "1"



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
    n = len(distribution_x)
    m = len(index_red)
    index_reduced = index_red.copy()
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


def loc_shuffle(arr):
    #fisher-yates algorithm
    n = len(arr)

    for i in range(n-1, 0, -1):
        j = random.randint(0, i)
        arr[i], arr[j] = arr[j], arr[i]

    return arr


def local_search_ff_random(distribution_x, index_red, l):

    n = len(distribution_x)
    m = len(index_red)
    index_reduced=index_red.copy()
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
    improvement = True

    while improvement:
        best_i = -1
        best_j = -1
        best_m = []

        random.shuffle(index_reduced)

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



data = generate_data(100)
mm=[m for m in range(10,90)]

temps = [0]*len(mm)
temps_random_mean=[0]*len(mm)
temps_random_max=[0]*len(mm)
temps_random_min=[0]*len(mm)

for i in range(len(mm)):
    print(i,"/80")
    dup = dupacova_forward(data[0],mm[i],2,data[1])[0]
    dup_start= set_to_index(dup,data[0])

    t = time.time()
    a = local_search_ff(data[0],dup_start,2)
    t1=time.time()
    temps[i]=t1-t


    iter_random=25
    temps_i_random=[0]*iter_random

    for k in range(iter_random):
        t1=time.time()
        b = local_search_ff_random(data[0],dup_start,2)
        t2 = time.time()
        temps_i_random[k]=t2-t1

    temps_random_max[i]=max(temps_i_random)
    temps_random_min[i]=min(temps_i_random)
    temps_random_mean[i]=sum(temps_i_random)/iter_random

plt.figure(figsize=(10, 6))

plt.plot(mm, temps, label="deterministic first-fit", color='black')
plt.plot(mm, temps_random_mean, label="mean random first-fit ", color='blue')
plt.plot(mm, temps_random_max, label="worst case random first-fit", color='red',linestyle=':')
plt.plot(mm, temps_random_min, label="best case random first-fit", color='green', linestyle=':')
plt.xlabel('m')
plt.title("Comparison of first-fit methods, n=100")
plt.legend()
plt.show()