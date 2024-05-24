import numpy as np
import math
import random
from scipy.optimize import linprog
import matplotlib.pyplot as plt

random.seed(11)

def norm_sq(x,y):
    value=0.
    n=len(x)
    for i in range(n):
        value+=(x[i]-y[i])**2
    return value

def matrice_distance(distribution_1,distribution_2):
    n_i=len(distribution_1)
    n_j=len(distribution_2)
    matrice=np.zeros((n_i,n_j))
    for i in range(n_i):
        for j in range(n_j):
            #essayer d'appeler norm plutôt
            matrice[i,j]=norm_sq(distribution_1[i],distribution_2[j])
    return matrice

def vecteur_contrainte(distribution_1_p,distribution_2_p):
    return distribution_1_p+distribution_2_p

def matrice_contrainte(n,m):
    matrice_contrainte=[]
    for i in range(n):
        a=np.zeros(n*m)
        for j in range(m):
            a[i*m+j]=1
        matrice_contrainte.append(a)
    for j in range(m):
        a=np.zeros(n*m)
        for i in range(n):
            a[m*i+j]=1
        matrice_contrainte.append(a)
    return matrice_contrainte


def vecteur_cout(n,m,distance):
    vect=[]
    for i in range(n):
        for j in range(m):
            a= distance[i,j]
            vect.append(a)
    return vect

def d_wasserstein(distribution_1_x,distribution_1_p,distribution_2_x,distribution_2_p):

    n = len(distribution_1_p)
    m = len(distribution_2_p)

    #il faut définir une forme pour la matrice pi(i,j) et ensuite utiliser ce qu'on a fait.
    #on peut écrire vecteur coût : (pi(1,1),pi(1,2),...,pi(1,n_j),...)

    matrice=matrice_distance(distribution_1_x,distribution_2_x)
    vecteur_cout_= vecteur_cout(n,m,matrice)
    vecteur_contrainte_=vecteur_contrainte(distribution_1_p, distribution_2_p)
    matrice_contrainte_=matrice_contrainte(n,m)

    # Définir la fonction objective et les contraintes sous forme canonique
    c = vecteur_cout_  # fonction objective à minimiser
    A = matrice_contrainte_  # matrice des contraintes
    b = vecteur_contrainte_  # vecteur des contraintes

    # Résoudre le problème d'optimisation linéaire
    res = linprog(c, A_eq=A, b_eq=b, method='highs')


    #print('Solution optimale:')
    #print('x:', res.x)
    #print('Valeur de la distance de Wasserstein:', res.fun)

    pi=res.x
    value = res.fun

    return (value,pi)

def dupacova(distribution_x,distribution_p,m):
    #m is the reduced number of atoms
    n = len(distribution_p)
    eps=0.0001

    if (m>n-1):
        print("choose a different m such that m<n")
        return 0

    elif (abs(1-sum(distribution_p))>eps):
        print("have a look on probabilities, sum(proba)=1")
        return 0

    else:

        atoms_chosen=[]
        atoms_to_chose=distribution_x.copy()

        for i in range(m):

            distance_w=np.zeros(n-i)

            for j in range(n-i): #n-i is equal to len(atoms_to_chose)
                distribution_dupacova_x=atoms_chosen.copy() #reinitialize each iteration
                distribution_dupacova_x.append(atoms_to_chose[j])
                #define probabilities of this dupacova distribution
                distribution_dupacova_p=[0]*len(distribution_dupacova_x)
                matrice__= matrice_distance(distribution_x,distribution_dupacova_x)
                for k in range(n):
                    closest=np.argmin(matrice__[k])
                    distribution_dupacova_p[closest]+=1

                for k in range(i+1):
                    distribution_dupacova_p[k]/=n


                distance_w[j]=d_wasserstein(distribution_x,distribution_p,distribution_dupacova_x,distribution_dupacova_p)[0]

            atoms_chosen.append(atoms_to_chose[np.argmin(distance_w)])
            atoms_to_chose.remove(atoms_to_chose[np.argmin(distance_w)])

        #now that we've selected atoms
        distribution_dupacova_final_x=atoms_chosen
        distribution_dupacova_final_p=np.zeros(m)
        matrice_distance_final=matrice_distance(distribution_x,distribution_dupacova_final_x)

        for k in range(n):
            closest=np.argmin(matrice_distance_final[k])
            distribution_dupacova_final_p[closest]+=1
        for k in range(m):
            distribution_dupacova_final_p[k]/=n

        return (distribution_dupacova_final_x,distribution_dupacova_final_p.tolist())

def kmeans(distribution_x,k):

    n=len(distribution_x)
    if k>n-1:
        print("choose another k, remember k<n")
        return 0
    else:
        d=len(distribution_x[0]) #dimension
        atoms_i_1=distribution_x[:k] #atoms before iteration
        atoms_i=[] #atoms after iteration
        iter=0

        while atoms_i != atoms_i_1 and iter<500:
            atoms_i = atoms_i_1
            iter += 1
            #we compute the partition
            distance=matrice_distance(distribution_x,atoms_i)
            partition=[[] for z in range(k)]
            for i in range(n):
                argmin=np.argmin(distance[i])
                partition[argmin].append(i)
            #when l=2 it is easy to find the argmin, it is the mean, now we update the atoms
            for l in range(k):
                mean=[0]*d
                for x in partition[l]:
                    for y in range(d):
                        mean[y]+=distribution_x[x][y]
                for y in range(d):
                    mean[y]=mean[y]/(len(partition[l]))
                atoms_i_1[l]=mean

        return atoms_i_1

def get_p(distribution_x,reduced_x):
    n=len(distribution_x)
    m=len(reduced_x)
    distance=matrice_distance(distribution_x,reduced_x)
    reduced_p=[0]*len(reduced_x)
    for k in range(n):
        closest=np.argmin(distance[k])
        reduced_p[closest]+=1
    for k in range(m):
        reduced_p[k]/=n
    return reduced_p

def local_search(distribution_x,distribution_p,m):
    #best-first
    n=len(distribution_x)

    if m>n-1:
        print("choose another m, remember m<n")
        return 0

    else:
        reduced_set=[] #before iteration
        reduced_set_1=distribution_x[:m].copy() #after iteration
        not_used=distribution_x[m:].copy()
        distance_to_reduce=d_wasserstein(distribution_x,distribution_p,reduced_set_1,get_p(distribution_x,reduced_set_1))[0]

        while reduced_set != reduced_set_1:
            #compute the distance we aim to reduce
            reduced_set=reduced_set_1.copy()
            reduced_set_p=get_p(distribution_x,reduced_set)
            best_first=np.zeros((m,n-m)) #matrix computing the distances in order to choose the best swap
            for i in range(m):
                for j in range(n-m):
                    reduced_set_11=reduced_set_1.copy()
                    reduced_set_11[i]=not_used[j]
                    reduced_set_11_p=get_p(distribution_x,reduced_set_11)
                    best_first[i,j]=d_wasserstein(distribution_x,distribution_p,reduced_set_11,reduced_set_11_p)[0]

            best_d=np.min(best_first)

            if best_d<distance_to_reduce:
                index=np.argmin(best_first)
                i = index // (n-m)
                j = index - i*(n-m)
                temp=not_used[j]
                not_used[j]=reduced_set_1[i]
                reduced_set_1[i]=temp
                distance_to_reduce=best_d
            else:
                reduced_set_1=reduced_set #exit the loop


        return (reduced_set_1,get_p(distribution_x,reduced_set_1))


m= 15
n= 30

distribution = [[random.uniform(0, 10) for _ in range(5)] for _ in range(n)]

distribution_p=[1/30 for _ in range(n)]

aa=dupacova(distribution,distribution_p,m)

print("avec dupacova on obtient une distance de ")

print(d_wasserstein(distribution,distribution_p,aa[0],aa[1])[0])
#kmeans(distribution_1_x_,2)

a=kmeans(distribution_2,m)
print("avec kmeans, on obtient")
print(d_wasserstein(distribution,distribution_p,a,get_p(distribution_2,a))[0])

aaa=local_search(distribution,distribution_p,m)
print("avec local search on obtient une distance de ")
print(d_wasserstein(distribution,distribution_p,aaa[0],aaa[1])[0])