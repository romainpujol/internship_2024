#######################################################################
# Problem-dependent costs
#######################################################################

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy.optimize import linprog


np.random.seed(618898)

def generate_data(n):
    # generate scenarios
    samples_normal = np.random.normal(10, 3, size=n)
    samples_normal2 = np.random.normal(10, 3, size=n)
    samples_gamma = np.random.gamma(5, 2, size=n)
    samples_gamma2 = np.random.gamma(3, 2, size=n)
    samples_gamma3 = np.random.gamma(2, 2, size=n)
    samples_uni = np.random.uniform(0, 10, size=n)

    samples = []
    for i in range(n):
        sample = np.array([samples_normal[i], samples_normal2[i], samples_gamma[i], samples_gamma2[i], samples_gamma3[i], samples_uni[i]])
        samples.append(sample)

    return np.array(samples)


def gradient(x,data,c):
    #gradient at x for a specific scenarios named data
    xi=np.array(data)
    return np.array(c)+2*np.dot(xi,xi)*(np.array(x))+xi

def generate_mini_batch(m,data):
    # generate a mini batch from the whole data matrix, you choose m lines
    # in the whole matrix
    n=len(data)
    data_copy=data.copy()
    if 5*m>n:
        print("take a smaller m, this is not a mini batch")
        return []
    else:
        random.shuffle(data_copy)
        return data_copy[:m]

def optimal_x(data,c):
    #for this particular problem, there is a closed formula in order to compute x*
    n=len(data)

    x = np.array([0,0,0,0,0,0])
    sum_xi = np.array([0,0,0,0,0,0])
    sum_norm=0

    for i in range(n):
        sum_xi = sum_xi+np.array(data[i])
        sum_norm += np.dot(data[i],data[i])

    x_opt= -(n/2)*(c + sum_xi/n)/sum_norm
    return x_opt


def optimal_x_batch(data,m,n_iter,c):
    # find an approximate x* for a mini-batch using SGD
    mini_batch=generate_mini_batch(m,data)
    x = np.array([0,0,0,0,0,0])

    for i in range(n_iter):
        step=1/(i+1)
        k = random.randint(0,m-1) #randomly choose the gradient
        x =x- step*gradient(x,mini_batch[k],c)

    return x


def cost_matrix(data,c):
    #cost matrix with the new metric
    n= len(data)
    m= n//5
    n_iter = 1000
    x = optimal_x_batch(data,m,n_iter,c)
    #opt_x = optimal_x(data,c)
    #print("distance euclidienne entre les optimum:", np.dot(x-opt_x,x-opt_x))
    norm_x_2 = np.dot(x,x)
    cost=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            cost[i,j]=abs(norm_x_2*(np.dot(data[i],data[i])-np.dot(data[j],data[j]))+np.dot(data[i]-data[j],x))

    return cost

def set_to_index(reduced, big):
    indices = []
    for i in range(len(reduced)):
        for j in range(len(big)):
            if np.array_equal(reduced[i], big[j]):
                indices.append(j)
                break
    return indices

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


def dupacova_forward_sgd(data,m,c):

    D = cost_matrix(data,c) #problem dependent
    n = len(data)
    minimum = [1000000000]*n
    reduced_set = []
    index_to_chose=[i for i in range(n)]
    best_d=10000000000

    while len(reduced_set)<m:
        for z in index_to_chose:
            minimum_i=minimum.copy()
            minimum_i=minimum_vector(minimum_i,D[z])
            distance=(1/n)*sum(minimum_i)
            if distance<best_d:
                index=z
                best_m=minimum_i
                best_d=distance
        minimum=best_m
        reduced_set.append(data[index])
        index_to_chose.remove(index)

    return np.array(reduced_set),D

def matrice_distance(distribution_1,distribution_2,l):
    n_i=len(distribution_1)
    n_j=len(distribution_2)
    matrice=np.zeros((n_i,n_j))
    for i in range(n_i):
        for j in range(n_j):
            matrice[i,j]=norm_l(distribution_1[i],distribution_2[j],l)
    return matrice

def norm_l(x,y,l):
    value=0.
    n=len(x)
    for i in range(n):
        value+=(x[i]-y[i])**2
    return value**(l/2)


def get_p(reduced,big,matrix):
    #get the optimal probabilities for a fixed set of scenarios
    m = len(reduced)
    n = len(big)
    numbers= [0]*m
    for i in range(n):
        numbers[np.argmin(matrix[i])]+=1
    for i in range(m):
        numbers[i]=numbers[i]/n
    return numbers


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
            distance=(1/n)*sum(minimum_i)
            if distance<best_d:
                index=i
                best_m=minimum_i
                best_d=distance
        minimum=best_m
        reduced_set.append(distribution_x[index])
        index_to_chose.remove(index)

    return reduced_set

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


def w_dist(costs,d1,p1,p2,d2):
    n1=len(p1)
    n2=len(p2)
    for i in range(n1):
        for j in range(n2):
            vecteur_cout.append(costs[i,j])
    matrice_contrainte=[]
    vecteur_contrainte=[]
    for i in range(n1):
        vecteur_contrainte.append(p1[i])
        a=np.zeros(n1*n2)
        for j in range(n2):
            a[i*n2+j]=1
        matrice_contrainte.append(a)

    for j in range(n2):
        vecteur_contrainte.append(p2[j])
        a=np.zeros(n1*n2)
        for i in range(n1):
            a[n2*i+j]=1
        matrice_contrainte.append(a)

    # Définir la fonction objective et les contraintes sous forme canonique
    c = vecteur_cout  # fonction objective à minimiser
    A = matrice_contrainte  # matrice des contraintes
    b = vecteur_contrainte  # vecteur des contraintes

    # Résoudre le problème d'optimisation linéaire
    res = linprog(c, A_eq=A, b_eq=b, method='highs')
    # x = res.x
    return res.fun


n = 1000
c = np.array([-5,-2,-3,0,-1,-2])*100
data = generate_data(n)

print(data[0:5])
m=[5,10,20,30,40,50]
ratio_to_opt = [0]*len(m)
ratio_to_opt_euc = [0]*len(m)
time_sgd = [0]*len(m)
time_euc = [0]*len(m)

x = optimal_x(data,c)
sss = 0
norm_x = np.dot(x,x)
for k in range(n):
    sss+= norm_x*np.dot(data[k],data[k])+np.dot(x,data[k])
v_opt = np.dot(c,x)+(1/n)*sss
num_runs = 10


ratios_sgd = np.zeros((len(m), num_runs))

for i in range(len(m)):
    print(m[i])
    for run in range(num_runs):
        #different batchs give different answers.
        t7 = time.time()
        dup_fw_sgd = dupacova_forward_sgd(data, m[i], c)

        reduced_sgd = dup_fw_sgd[0]
        index_sgd = set_to_index(reduced_sgd, data)

        matrix_sgd = (dup_fw_sgd[1])[index_sgd]
        get_p_sgd = get_p(reduced_sgd, data, np.transpose(matrix_sgd))

        opt_x_sgd = c
        sum_xi_norm = 0

        for k in range(m[i]):
            opt_x_sgd = opt_x_sgd + get_p_sgd[k] * reduced_sgd[k]
            sum_xi_norm += get_p_sgd[k] * np.dot(reduced_sgd[k], reduced_sgd[k])

        opt_x_sgd = (-1/2) * opt_x_sgd / sum_xi_norm

        norm_x_sgd = np.dot(opt_x_sgd, opt_x_sgd)
        opt_v_sgd = np.dot(opt_x_sgd, c)
        for k in range(m[i]):
            opt_v_sgd += get_p_sgd[k] * (norm_x_sgd * np.dot(reduced_sgd[k], reduced_sgd[k]) + np.dot(opt_x_sgd, reduced_sgd[k]))
        t8 = time.time()
        time_sgd[i] = t8 - t7

        ratios_sgd[i][run] = abs((opt_v_sgd - v_opt) / v_opt)

    t = time.time()
    reduced_euc = dupacova_forward(data, m[i], 2)
    index_euc = set_to_index(reduced_euc, data)
    matrix_euc = matrice_distance(data, reduced_euc, 2)
    get_p_euc = get_p(reduced_euc, data, matrix_euc)

    opt_x_euc = c
    sum_xi_norm_euc = 0

    for k in range(m[i]):
        opt_x_euc = opt_x_euc + get_p_euc[k] * reduced_euc[k]
        sum_xi_norm_euc += get_p_euc[k] * np.dot(reduced_euc[k], reduced_euc[k])

    opt_x_euc = (-1/2) * opt_x_euc / sum_xi_norm_euc

    norm_x_euc = np.dot(opt_x_euc, opt_x_euc)
    opt_v_euc = np.dot(opt_x_euc, c)
    for k in range(m[i]):
        opt_v_euc += get_p_euc[k] * (norm_x_euc * np.dot(reduced_euc[k], reduced_euc[k]) + np.dot(opt_x_euc, reduced_euc[k]))
    t2 = time.time()
    time_euc[i] = t2 - t

    ratio_to_opt_euc[i] = abs((opt_v_euc - v_opt) / v_opt)

avg_ratios_sgd = np.mean(ratios_sgd, axis=1)
min_ratios_sgd = np.min(ratios_sgd, axis=1)
max_ratios_sgd = np.max(ratios_sgd, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(m, avg_ratios_sgd, label="Average optimality ratio SGD",color='blue')
plt.plot(m, min_ratios_sgd, label="Best case optimality ratio SGD", linestyle='--',color='green')
plt.plot(m, max_ratios_sgd, label="Worst case optimality ratio SGD", linestyle='--',color='red')
plt.plot(m, ratio_to_opt_euc, label="Optimality ratio Euclidean",color='orange')

plt.xlabel('m')
plt.ylabel('ratio')
plt.legend()
plt.title("Optimality ratio, n=1000")
plt.show()
