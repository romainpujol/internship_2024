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
                # No improvement
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

def local_search_ff(distribution_x, index_red, l):
    n = len(distribution_x)
    m = len(index_red)
    index_reduced=index_red.copy()

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
                        break  # improvement
            if best_i != -1 and best_j != -1:
                break  # improvement

        if best_i == -1 and best_j == -1:
            improvement = False
        else:
            index_reduced.remove(best_i)
            index_reduced.append(best_j)
            minimum = best_m

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




n = 100
deb = 10

distribution=generate_data_u(n)

mm = [i for i in range(deb,n-deb)]

time_bf = [0]*len(mm)
time_ff = [0]*len(mm)
time_ff_dup_reverse= [0]*len(mm)
time_ff_dup = [0]*len(mm)
time_bf_dup_reverse= [0]*len(mm)
time_bf_dup= [0]*len(mm)
time_ff_km= [0]*len(mm)
time_bf_km=[0]*len(mm)

value_bf = [0]*len(mm)
value_ff = [0]*len(mm)
value_ff_dup_reverse= [0]*len(mm)
value_ff_dup = [0]*len(mm)
value_bf_dup_reverse= [0]*len(mm)
value_bf_dup= [0]*len(mm)
value_ff_km= [0]*len(mm)
value_bf_km= [0]*len(mm)

for i in range(len(mm)):
    print(i, "/", len(mm))
    #run of local-search best-fit
    t1=time.time()
    bf = local_search_bf(distribution[0],list(range(mm[i])),2)
    t2=time.time()
    time_bf[i]=t2-t1
    value_bf[i]=bf[1]

    #run of local-search first-fit
    t5=time.time()
    ff = local_search_ff(distribution[0],list(range(mm[i])),2)
    t6=time.time()
    time_ff[i]=t6-t5
    value_ff[i]=ff[1]

    #run dupacova in order to get the starter and then
    t3=time.time()
    dupacova = dupacova_forward(distribution[0],mm[i],2,distribution[1])
    t4=time.time()
    # add t4-t3 to the duration of bf and ff with dupacova starters
    index_dup=set_to_index(dupacova[0],distribution[0])
    index_dup_rev=index_dup[::-1]

    t7=time.time()
    ff_dup_rev = local_search_ff(distribution[0],index_dup_rev,2)
    t8=time.time()
    time_ff_dup_reverse[i]=t8-t7+t4-t3
    value_ff_dup_reverse[i]=ff_dup_rev[1]

    t9=time.time()
    ff_dup = local_search_ff(distribution[0],index_dup,2)
    t10=time.time()
    time_ff_dup[i]=t10-t9+t4-t3
    value_ff_dup[i]=ff_dup[1]

    t50=time.time()
    bf_dup = local_search_bf(distribution[0],index_dup,2)
    t51=time.time()
    time_bf_dup[i]=t51-t50+t4-t3
    value_bf_dup[i]=bf_dup[1]


    t11=time.time()
    bf_dup_rev = local_search_bf(distribution[0],index_dup_rev,2)
    t12=time.time()
    time_bf_dup_reverse[i]=t12-t11+t4-t3
    value_bf_dup_reverse[i]=bf_dup_rev[1]


    #run of a k-means
    t1_km=time.time()
    k=mm[i]
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(distribution[0])
    centroids = kmeans.cluster_centers_
    mat_dist = matrice_distance(centroids,distribution[0],2)
    centers = []

    for z in range(mm[i]):
        centers.append(np.argmin(mat_dist[z]))

    if len(set(centers))== mm[i]:
        t2_km=time.time()
        t15=time.time()
        bf_km=local_search_bf(distribution[0],centers,2)
        t16=time.time()
        value_bf_km[i] = bf_km[1]
        time_bf_km[i] = t16-t15+t2_km-t1_km
        t17=time.time()
        ff_km=local_search_ff(distribution[0],centers,2)
        t18=time.time()
        value_ff_km[i] = ff_km[1]
        time_ff_km[i] = t18-t17+t2_km-t1_km


plt.figure(figsize=(10, 6))
plt.plot(mm,time_bf,label="Best-fit")
plt.plot(mm,time_ff, label = "First-fit")
plt.plot(mm,time_bf_dup,label="Best-fit, Dupacova starters")
plt.plot(mm,time_ff_dup, label = "First-fit, Dupacova starters")
plt.plot(mm,time_bf_dup_reverse,label="Best-fit, reverse Dupacova starters")
plt.plot(mm,time_ff_dup_reverse, label = "First-fit, reverse Dupacova starters")
plt.plot(mm,time_bf_km,label="Best-fit, m-means neighboors starters")
plt.plot(mm,time_ff_km, label = "First-fit, m-means neighboors starters")

plt.xlabel('m')
plt.legend()
plt.title("Run time comparison, n=100")
plt.show()