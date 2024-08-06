###########################################################
# Comparison Dupacova algorithm: forward, backward
###########################################################

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB

np.random.seed(11112002)
random.seed(11112002)

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

n = 100
deb = 10

mm = [i for i in range(deb,n-deb)]

temps_forward=[0]*(len(mm))
temps_backward=[0]*(len(mm))
temps_milp=[0]*(len(mm))

distance_forward=[0]*(len(mm))
distance_backward=[0]*(len(mm))
distance_milp=[0]*(len(mm))

distribution,probabilities = generate_data(n)

for k in range(len(mm)):

    print(k,"/", len(mm))
    tp1=time.time()
    bb=dupacova_forward(distribution,mm[k],2,probabilities)
    tp2=time.time()
    bbb=dupacova_backward(distribution,mm[k],2,probabilities)
    tp3=time.time()
    temps_forward[k]=tp2-tp1
    distance_forward[k]=sum(bb[1])/n
    temps_backward[k]=tp3-tp2
    distance_backward[k]=sum(bbb[1])/n
    tp5=time.time()
    ccc= milp_formulation(distribution,mm[k])
    tp6=time.time()
    temps_milp[k]=tp6-tp5
    distance_milp[k]=ccc

plt.figure(figsize=(10, 6))

plt.plot(mm,distance_forward,label="Forward Dupačová")
plt.plot(mm,distance_backward, label = "Backward Dupačová")
plt.plot(mm,distance_milp,label="MILP formulation - Gurobi ")
plt.xlabel("n")
plt.legend()
#plt.title("Run time comparison, variable n and m=n/4")
plt.title("Efficiency in term of Wasserstein distance, Dupačová algorithms, n=100")
plt.show()

