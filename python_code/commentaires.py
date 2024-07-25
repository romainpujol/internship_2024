################################################################################
##################### Commentaires généraux en vrac ############################
################################################################################
"""
Je liste quelques remarques sur le code, j'ai mis plus bas quelques commentaires plus détaillés sur des points précis.

- Tu gagnerais à "factoriser" ton code sur certains points : tu peux faire un fichier utils.py, et y mettre des fonctions redondantes comme norm_l dedans. Comme ça quand elle a un problème (cf. ci-dessous), tu as juste à changer celle du fichier. Bon pour cet exemple précis de norm_l, tu peux utiliser la fonction de numpy (cf. ci-dessous aussi), ça va plus vite que la coder à la main en Python.
- Plus généralement, essayer de "factoriser" ton code quand tu dois écrire la même fonction plusieurs fois, ça permet d'aller plus vite. (C'est un vrai terme en programmation, les gens parlent de "refactor" un code après un premier jet pour éliminer ce genre de redondances).

"""
################################################################################
################### Commentaire sur norm_l et numpy ############################
################################################################################

"""
Ta fonction norm_l est buguée
def norm_l(x,y,l):
    value=0.
    n=len(x)
    for i in range(n):
        value+=(x[i]-y[i])**2
    return value**(l/2)

Ca devrait être return value**(1/l) et (x[i]-y[i])**l.

De plus tu pourrais utiliser la fonction norm de numpy pour aller plus vite : cf le petit ex ci-dessous. L'un avec la boucle for nativement en Python, le second avec np.dot, la troisième avec np.linalg.norm. 

C'est un peu le travers et la force de Python : en soit le language est nul pour faire des maths mais numpy fait passer beaucoup de fonctions communes à des languages plus bas niveau et du coup ça va plus vite. Le revers de la médaille c'est qu'il faut au maximum utiliser ces packages qui utilisent des routines bas niveau sinon en Python pur on se paye des surcoûts inexplicables mathématiquement parlant.

Dans notre cas, tu as sûrement déjà entendu des gens dire qu'il "faut pas faire de for" et travailler élément par élément mais qu'il faut "vectoriser son code" (et utiliser numpy).

Si ça t'intéresse, ci-dessous ce que j'ai compris du pourquoi. Il y a essentiellement trois choses :

1) Une loop for en Python c'est plus couteux que dans la plupart des languages bas niveau : Python est un language "interprété" vs language "compilé". Quand on lance le code Python ça lit ligne par ligne ton code à l'execution et le compilateur essaye de comprendre au fur et à mesure. Alors que dans un language compilé comme le C, ça pré-mache le travail à l'ordi. Faire une boucle for en Python c'est de base "anormalement" couteux comparé à ce à quoi on s'attend. Grosso modo, numpy ça utilise des fonctions en C (via Cython) ou en Fortran pour aller plus vite sur beaucoup de routines.  Mais ici numpy ne se contente pas de juste faire une boucle for en C, cf ce qui suit. 

2) Si ta collection de nombre est stockée contiguement en mémoire (un seul bloc), alors on peut exploiter ça. C'est ce que fait np.random.rand(...), les containers de np sont, comme en C, stockés contiguement en mémoire. Au contraire d'une liste de nombres en Python.

3) A ce moment, au lieu de faire une boucle parcourant les éléments du vecteur on peut, avec un language plus bas niveau, paralléliser ce morceau. C'est là le vrai gain. Pour cette parrallélisation, Numpy utilise un package bas niveau appelé BLAS si tu veux des détails https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms . 
"""

import numpy as np
import time

# Norm with native for loop
def norm_l(x, y, l):
    value = 0.
    n = len(x)
    for i in range(n):
        value += (x[i] - y[i])**l
    return value**(1/l)

def norm_2(x,y):
    diff = np.array(x) - np.array(y)
    return np.dot(diff, diff)

# NumPy-based norm function
def norm_l_numpy(x, y, l):
    return np.linalg.norm(np.array(x) - np.array(y), ord=l)

# Generate random vectors
x = np.random.rand(3000000)
y = np.random.rand(3000000)

# Custom function
start_time = time.time()
custom_norm = norm_l(x, y, 2)
custom_duration = time.time() - start_time
print(f"Custom function result: {custom_norm}, Time taken: {custom_duration} seconds")

# NumPy with np.dot
start_time = time.time()
numpy_norm = norm_l_numpy(x, y, 2)
numpy_duration = time.time() - start_time
print(f"NumPy dot result: {numpy_norm}, Time taken: {numpy_duration} seconds")

# NumPy with np.linalg.norm
start_time = time.time()
numpy_norm = norm_l_numpy(x, y, 2)
numpy_dot = time.time() - start_time
print(f"NumPy norm result: {numpy_norm}, Time taken: {numpy_dot} seconds")

##############################