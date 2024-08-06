################################################################################
####################### Commentaires généraux en vrac ##########################
################################################################################
"""
Dans le rapport 
- il faut pour le thm 17 qui donne la redistribution optimale
des poids, donner aussi la formule fermée pour la valeur. Ajuster la preuve en
conséquence pour marquer le coup quand elle est prouvée.
- Algo 2 préciser "best"
- Algo 3, mauvaise parenthèse : ça devrait être (supp(P) \ R) x R
- Algo 3, on a 
    min_{i' \in I \ {i}} c(x_k, x_i') = 
                                max( min_{i' \in I}c(x_k, x_i'), c(x_k, x_i) )

Je liste quelques remarques sur le code, j'ai mis plus bas quelques commentaires
Splus détaillés sur des points précis.

Sur c_l approximation comparison.py 
- Tu gagnerais à "factoriser" ton code sur
certains points : tu peux faire un fichier utils.py, et y mettre des fonctions
redondantes comme norm_l dedans. Comme ça quand elle a un problème (cf.
ci-dessous), tu as juste à changer celle du fichier. Bon pour cet exemple précis
de norm_l, tu peux utiliser la fonction de numpy (cf. ci-dessous aussi), ça va
plus vite que la coder à la main en Python.

 - Quelques cas où tu peux
utiliser les fonctions de numpy.

- Toujours dans l'idée d'utiliser au mieux les fonctionnalités de numpy (et la
  parallélisation d'opérations vectorielles), il faut viser à faire le moins
  possible de boucles for. Un exemple : 
      for i in range(n):
        d = float('inf')
        for k in index_reduced:
            dist = D[i][k]
            if dist < d:
                d = dist
                index_closest[i] = k
        minimum[i] = d
    Tu peux vectoriser la boucle interne et obtenir
        for i in range(n):
            k = np.argmin(D[i])
            minimum[i] = D[i][k]
    c'est un peu plus délicat pour la boucle externe mais c'est possible aussi :
        index_closest = np.argmin(D, axis=1)
        minimum = D[np.arange(n), index_closest]
    où np.arrange(n) ça crée une np.array de 0 à n-1. On en profite pour 
    définir index_closest et minimum directement comme ça.

    Le gain est considérable en temps d'éxecution, cf la remarque sur norm_l et
    numpy plus bas, c'est la même idée.

 - Dans dupacova_forward, l'opération
index_to_chose.remove(index) est O(n) car ton conteneur pour les index est une
liste. C'est O(1) si le conteneur était un set. (En Python un set c'est un
hashset alors qu'une list il faut la parcourir pour trouver la valeur
correspondante avant de la supprimer. Ce n'est pas grave vu que l'algo est
O(mn²)), mais c'est une petite amélioration. 

"""
################################################################################
################### Commentaire sur norm_l et numpy ############################
################################################################################

"""
Ta fonction norm_l est buguée def norm_l(x,y,l):S
    value=0. n=len(x) for i in range(n):
        value+=(x[i]-y[i])**2
    return value**(l/2)

Ca devrait être return value**(1/l) et (x[i]-y[i])**l.

De plus tu pourrais utiliser la fonction norm de numpy pour aller plus vite : cf
le petit ex ci-dessous. L'un avec la boucle for nativement en Python, le second
avec np.dot, la troisième avec np.linalg.norm. 

C'est un peu le travers et la force de Python : en soit le language est nul pour
faire des maths mais numpy fait passer beaucoup de fonctions communes à des
languages plus bas niveau et du coup ça va plus vite. Le revers de la médaille
c'est qu'il faut au maximum utiliser ces packages qui utilisent des routines bas
niveau sinon en Python pur on se paye des surcoûts inexplicables
mathématiquement parlant.

Dans notre cas, tu as sûrement déjà entendu des gens dire qu'il "faut pas faire
de for" et travailler élément par élément mais qu'il faut "vectoriser son code"
(et utiliser numpy).

Si ça t'intéresse, ci-dessous ce que j'ai compris du pourquoi. Il y a
essentiellement trois choses :

1) Une loop for en Python c'est plus couteux que dans la plupart des languages
   bas niveau : Python est un language "interprété" vs language "compilé". Quand
   on lance le code Python ça lit ligne par ligne ton code à l'execution et le
   compilateur essaye de comprendre au fur et à mesure. Alors que dans un
   language compilé comme le C, ça pré-mache le travail à l'ordi. Faire une
   boucle for en Python c'est de base "anormalement" couteux comparé à ce à quoi
   on s'attend. Grosso modo, numpy ça utilise des fonctions en C (via Cython) ou
   en Fortran pour aller plus vite sur beaucoup de routines.  Mais ici numpy ne
   se contente pas de juste faire une boucle for en C, cf ce qui suit. 

2) Si ta collection de nombre est stockée contiguement en mémoire (un seul
   bloc), alors on peut exploiter ça. C'est ce que fait np.random.rand(...), les
   containers de np sont, comme en C, stockés contiguement en mémoire. Au
   contraire d'une list en Python.

3) A ce moment, au lieu de faire une boucle parcourant les éléments du vecteur
   on peut, avec un language plus bas niveau, paralléliser ce morceau. C'est là
   le vrai gain. Pour cette parallélisation, numpy utilise un package bas
   niveau appelé BLAS si tu veux des détails
   https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms . 
"""

import numpy as np
import time

# Norm with native for loop
def norm_2(x, y):
    value = 0.
    n = len(x)
    for i in range(n):
        value += (x[i] - y[i])**2
    return value**(1/2)

# Using np.dot
def norm_2_dot(x,y):
    diff = np.array(x) - np.array(y)
    return np.sqrt(np.dot(diff, diff))

# Using np.sum
def norm_2_sum(x, y):
    return np.sqrt(np.sum((x - y)**2))

# Using np.linalg.norm
def norm_2_numpy(x, y):
    return np.linalg.norm(np.array(x) - np.array(y)) 

# Generate random vectors
x = np.random.rand(3000000)
y = np.random.rand(3000000)

# Custom function
start_time = time.time()
custom_norm = norm_2(x, y)
custom_duration = time.time() - start_time
print(f"Custom function result: {custom_norm}, Time taken: {custom_duration} seconds")

# NumPy with np.dot
start_time = time.time()
numpy_dot = norm_2_dot(x, y)
numpy_dot_duration = time.time() - start_time
print(f"NumPy dot result......: {numpy_dot}, Time taken: {numpy_dot_duration} seconds")

# NumPy with np.sum
start_time = time.time()
numpy_sum = norm_2_sum(x, y)
numpy_sum_duration = time.time() - start_time
print(f"NumPy sum result......: {numpy_sum}, Time taken: {numpy_sum_duration} seconds")

# NumPy with np.linalg.norm
start_time = time.time()
numpy_norm = norm_2_numpy(x, y)
numpy_norm_duration = time.time() - start_time
print(f"NumPy norm result.....: {numpy_norm}, Time taken: {numpy_norm_duration} seconds")

"""
Custom function result: 707.1205039521882, Time taken: 1.062058687210083 seconds
NumPy dot result......: 707.1205039521776, Time taken: 0.027849912643432617
seconds NumPy sum result......: 707.1205039521778, Time taken:
0.023437023162841797 seconds NumPy norm result.....: 707.1205039521776, Time
taken: 0.029018878936767578 seconds

C'est donc essentiellement pareil avec les différentes variantes de numpy qui
sont ici significativement plus rapides que la version en Python pur.
"""

################################################################################
################### Commentaire sur norm_l et numpy ############################
################################################################################
