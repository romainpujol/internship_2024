###########################################################
# Visualization Dupacova algorithm: forward, backward
###########################################################

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance


np.random.seed(11112002)
random.seed(11112002)

def generate_data(n):

    x = np.random.normal(loc=10, scale=2, size=n)

    y = np.random.gamma(shape=2, scale=2, size=n)

    # Normalisation des probabilités pour que leur somme soit égale à 1
    probabilities = [1/n]*n

    data=[0]*n

    for i in range(n):
        data[i]=[x[i],y[i]]
    return (data,probabilities)

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

def dupacova_backward(distribution_x,m,l,distribution_p):
    D = matrice_distance(distribution_x,distribution_x,l)
    n = len(distribution_x)
    minimum = [0]*n
    reduced_set = distribution_x.copy()
    index_to_rm=[i for i in range(n)]
    index_closest=[i for i in range(n)]
    if len(distribution_p)==0:
        distribution_p = [1/n]*n


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
                d=sum_p(minimum_i,distribution_p)
            if d<d_best:
                d_best=d
                to_rm=i
                m_best=minimum_i
                best_index=index_closest_i

        minimum=m_best
        index_closest=best_index
        atom_to_rm=distribution[to_rm]
        reduced_set.remove(atom_to_rm)
        index_to_rm.remove(to_rm)

    return (reduced_set,minimum)


n = 50

temps_forward=[0]*(n-1)
temps_backward=[0]*(n-1)

distance_forward=[0]*(n-1)
distance_backward=[0]*(n-1)

distribution,probabilities = generate_data(n)


f=dupacova_forward(distribution,10,2,probabilities)[0]
b=dupacova_backward(distribution,10,2,probabilities)[0]

x_f=[]
y_f=[]
x_b=[]
y_b=[]

for i in range(10):
    x_f.append(f[i][0])
    y_f.append(f[i][1])
    x_b.append(b[i][0])
    y_b.append(b[i][1])

F = matrice_distance(distribution,f,2)
B = matrice_distance(distribution,b,2)

x=[]
y=[]
for i in range(n):
    x.append(distribution[i][0])
    y.append(distribution[i][1])

#visualization


plt.figure(figsize=(10, 6))


colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']


matrice = matrice_distance(distribution, f, 2)

for i in range(n):
    if distribution[i] not in f:
        closest = np.argmin(matrice[i])
        plt.scatter(distribution[i][0], distribution[i][1], color=colors[closest])

for i in range(10):
    plt.scatter(x_f[i],y_f[i],c=colors[i],marker='+',s=100)

plt.xlabel('x')
plt.ylabel('y')
plt.title("Forward Dupačová")

plt.show()
