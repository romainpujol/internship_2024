#######################################################################
# Multiproduct assembly problem
#######################################################################

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
import gurobipy as gp
from gurobipy import GRB

random.seed(11112002)
np.random.seed(11112002)


def scenarios_binomial(n):

    scen = []

    for i in range(n):
        d_1 = np.random.binomial(40,0.5)
        d_3 = np.random.binomial(40,0.5)
        d_i = [d_1, 4*d_1+random.randint(0,25), d_3,4*d_3 + random.randint(0,25), np.random.binomial(40,0.5), np.random.binomial(40,0.5), np.random.binomial(40,0.5), np.random.binomial(40,0.5), np.random.binomial(40,0.5), np.random.binomial(40,0.5)]
        scen.append(d_i)

    return (scen,[1/n]*n)


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

def dupacova_forward(distribution_x,m,l):
    #we assume that the distribution is uniform as it is meant to come from sampling.
    D = matrice_distance(distribution_x,distribution_x,l)
    n = len(distribution_x)
    minimum = [1000000000]*n
    reduced_set = []
    index_to_chose=[i for i in range(len(distribution_x))]
    best_d=10000000000
    index_chosen=[]

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

    prob = get_p(reduced_set,distribution_x,l)
    return (reduced_set,sum(minimum)/n,prob)

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

        prob = get_p(reduced_distribution,distribution_x,l)
        return (reduced_distribution, sum(minimum) / n,prob)

def get_p(reduced,big,l):
    m = len(reduced)
    n = len(big)
    D = matrice_distance(big,reduced,l)
    numbers= [0]*m
    for i in range(n):
        numbers[np.argmin(D[i])]+=1
    for i in range(m):
        numbers[i]=numbers[i]/n
    return numbers

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

def solve_mpa(scen,A,q,l,c,s,prob):

    n_products = len(q)
    n_sub = len(c)
    K = len(scen)

    vec_l_q = []#l-q
    for i in range(n_products):
        vec_l_q.append(l[i]-q[i])

    model = gp.Model("model")
    model.setParam('OutputFlag', 0)

    x = model.addVars(n_sub, vtype=GRB.CONTINUOUS, name="x")
    y = model.addVars(n_sub,K,vtype=GRB.CONTINUOUS, name="y")
    z = model.addVars(n_products,K,vtype=GRB.CONTINUOUS, name ="z")

    if len(prob)==0:
        #uniform
        model.setObjective(
            sum(c[f]*x[f] for f in range(n_sub)) +
            (1/K) * sum(vec_l_q[i] * z[i,k] for k in range(K) for i in range(n_products)) -
            (1/K) * sum(s[w] * y[w,m] for w in range(n_sub) for m in range(K)),
            GRB.MINIMIZE
            )
    else:
        model.setObjective(
            sum(c[f]*x[f] for f in range(n_sub)) +
            sum(prob[k]*vec_l_q[i] * z[i,k] for k in range(K) for i in range(n_products)) -
            sum(prob[m]*s[w] * y[w,m] for w in range(n_sub) for m in range(K)),
            GRB.MINIMIZE
            )

    model.addConstrs(y[j,k]==x[j]-sum(A[i][j]*z[i,k] for i in range(n_products)) for k in range(K) for j in range(n_sub))
    model.addConstrs(z[i,k] >= 0 for i in range(n_products) for k in range(K))
    model.addConstrs(z[i,k] <= scen[k][i] for i in range(n_products) for k in range(K))
    model.addConstrs(y[i,k] >= 0 for i in range(n_sub) for k in range(K))
    model.optimize()

    a = model.objVal
    x_value = {j: x[j].X for j in range(n_sub)}
    return a,x_value




A = np.array([[1,4,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,4,8,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,4,16,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,4,8,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,16,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0],[0,4,16,0,0,0,0,0,0,0,0,1,2,1,0,0,0,0,0],
[0,0,32,0,0,0,1,0,0,0,0,0,0,0,2,5,1,2,1],[0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,4,0,2,1],
[0,0,16,0,0,0,0,0,0,0,0,0,0,0,0,5,0,2,1],[0,0,16,0,0,0,0,0,0,0,0,0,0,0,2,4,0,2,1]])

q = np.array([300,100,350,120,500,200,300,70,50,50])
l = np.array([0,0,0,0,0,0,0,0,0,0])
c = np.array([100,5,0.5,5,5,10,50,20,20,30,30,10,10,20,30,10,10,20,20])
s = c/5

n = 250
m = [k for k in range(25,126)]

time_dup = [0]*len(m)
time_loc = [0]*len(m)

value_dup = [0]*len(m)
value_loc = [0]*len(m)

dist_dup=[0]*len(m)
dist_loc = [0]*len(m)

scenars = scenarios_binomial(n)

t= time.time()
z = solve_mpa(scenars[0],A,q,l,c,s,[])
t2= time.time()
time_milp = [t2-t]*len(m)
value_milp=[z[0]]*len(m)

for i in range(len(m)):

    print(i,"/",len(m))

    reduced = dupacova_forward(scenars[0],m[i],2)
    loc = local_search_bf(scenars[0],set_to_index(reduced[0],scenars[0]),2)

    dist_dup[i]=reduced[1]
    dist_loc[i]=loc[1]

    t2=time.time()
    now = solve_mpa(reduced[0],A,q,l,c,s,reduced[2])
    t3=time.time()
    value_dup[i]=now[0]
    time_dup[i]=t3-t2
    t4=time.time()
    now = solve_mpa(loc[0],A,q,l,c,s,loc[2])
    t5=time.time()
    value_loc[i]=now[0]
    time_loc[i]=t5-t4



ecart_dup = [abs((value_milp[i]-value_dup[i])/value_milp[i]) for i in range(len(value_milp))]
ecart_loc = [abs((value_milp[i]-value_loc[i])/value_loc[i]) for i in range(len(value_milp))]

plt.figure(figsize=(10, 6))
plt.plot(m,ecart_dup,label="Forward Dupacova")
plt.plot(m,ecart_loc, label = "Local-search")

plt.xlabel('m')
plt.ylabel('Wasserstein distance')
plt.legend()
plt.title("Distance to the original distribution, n=250")
plt.show()
