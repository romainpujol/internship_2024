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

os.environ["OMP_NUM_THREADS"] = "1"

        
random.seed(11112002)
np.random.seed(11112002)

# Dans numpy, de diverses façons possibles, cf commentaires.py
def norm_l(x,y,l):
    value=0.
    n=len(x)
    for i in range(n):
        value+=(x[i]-y[i])**2
    return value**(l/2)

# ajouté dans utils.py en tant que init_distanceMatrix
def matrice_distance(distribution_1,distribution_2, l):
    n_i=len(distribution_1)
    n_j=len(distribution_2)
    matrice=np.zeros((n_i,n_j))
    for i in range(n_i):
        for j in range(n_j):
            matrice[i,j]=norm_l(distribution_1[i],distribution_2[j],l)
    return matrice

# Dans numpy, np.minimum(v1,v2) 
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

# produit scalaire entre m et p, on peut utiliser np.dot(m,p)
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

# ajouté dans utils.py en tant que generate_data_normalgamma(n)
def generate_data_u(n):

    x = np.random.normal(loc=10, scale=2, size=n)

    y = np.random.gamma(shape=2, scale=2, size=n)

    probabilities = [1/n]*n

    data=[0]*n
    for i in range(n):
        data[i]=[x[i],y[i]]

    return (data,probabilities)

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

        tmp = 0
        while improvement:
            tmp += 1
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
                # print(f"{best_i} {best_j} {distance_to_reduce}")
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

# Now obsolete but need to do the opposite "index_to_set" function instead
def set_to_index(reduced,big):
    index=[]
    for i in range(len(reduced)):
        for j in range(len(big)):
            if np.array_equal(reduced[i],big[j]):
                index.append(j)
    if len(set(index))!=len(reduced):
        print("error")
        return 0
    else:
        return index

def sum_vec(v1,v2):
    if len(v1) != len(v2):
        print("error")
        return 0
    else:
        v3 = [0]*len(v1)
        for i in range(len(v1)):
            v3[i]=v1[i]+v2[i]
        return v3

def scalar_vec(v1,beta):
    for i in range(len(v1)):
        v1[i]=v1[i]*beta
    return v1


def local_search_improved(distribution,m):
    l=2
    eps = 10**(-3)
    d = len(distribution[0])
    n = len(distribution)
    #get the starters
    dupacova_starters = dupacova_forward(distribution,m,l,[1/n]*n)[0]
    index_starters = set_to_index(dupacova_starters,distribution)
    #first run of local-search
    ls = local_search_bf(distribution,index_starters,l)

    #update centers
    centers = ls[1]
    matrice=matrice_distance(distribution,centers,l)
    closest = [[] for _ in range(m)]

    # update centers
    for i in range(n):
        index_closest = np.argmin(matrice[i])
        closest[index_closest].append(i)
    new_centers=[[0]*d for _ in range(m)]
    for j in range(m):
        size = len(closest[j])
        for k in closest[j]:
            new_centers[j]=sum_vec(new_centers[j],distribution[k])
        new_centers[j]=scalar_vec(new_centers[j],1/size)

    matrice=matrice_distance(distribution,new_centers,l)
    dist = 0

    for x in range(n):
        dist += min(matrice[x])
    dist /= n

    return dist,centers

def k_means(distribution,m):

    n = len(distribution)

    km = KMeans(n_clusters=m)
    km.fit(distribution)
    centroids = km.cluster_centers_

    matrice = matrice_distance(distribution,centroids,2)
    dist = 0

    for i in range(n):
        dist +=(min(matrice[i]))

    return dist / n


# n =150
# deb = 20
# l = 2

# distribution=generate_data_u(n)
# mm = [i for i in range(deb,n-deb)]

# time_km = [0]*len(mm)
# time_ls = [0]*len(mm)
# time_ls_improved = [0]*len(mm)

# value_km= [0]*len(mm)
# value_ls= [0]*len(mm)
# value_ls_improved = [0]*len(mm)

# for i in range(len(mm)):
#     print(i, "/", len(mm) )
#     #get the starters
#     t1_starter = time.time()
#     dupacova_starters = dupacova_forward(distribution[0],mm[i],l,[1/n]*n)[0]
#     t2_starter=time.time()
#     index_starters = set_to_index(dupacova_starters,distribution[0])
#     t1=time.time()
#     value_ls[i]=local_search_bf(distribution[0],index_starters,l)[0]
#     t2=time.time()
#     value_ls_improved[i] =local_search_improved(distribution[0],mm[i])[0]
#     t3=time.time()
#     value_km[i]= k_means(distribution[0],mm[i])
#     t4=time.time()
#     time_km[i]=t4-t3
#     time_ls[i]=t2_starter-t1_starter + t2-t1
#     time_ls_improved[i]=t3-t2


# plt.figure(figsize=(10, 6))
# plt.plot(mm,value_km,label="k-means")
# plt.plot(mm,value_ls, label = "Local-search")
# plt.plot(mm,value_ls_improved,label="Local-search improved")

# plt.xlabel('m')
# plt.legend()
# plt.title("Efficiency comparison, n=150")
# plt.show()
