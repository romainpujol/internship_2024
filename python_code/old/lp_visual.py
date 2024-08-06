import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Définition du problème LP initial
c_initial = np.array([-1, -2])  # Coefficients de la fonction de coût (à maximiser)
c_modified = np.array([-1, -1.5])  # Coefficients légèrement modifiés
A = np.array([[1, 3], [2, 1], [1, 1.3]])  # Coefficients des contraintes
b = np.array([10, 6, 4])  # Limites des contraintes

# Résolution du problème initial
result_initial = linprog(c_initial, A_ub=A, b_ub=b, method='highs')

# Résolution du problème modifié
result_modified = linprog(c_modified, A_ub=A, b_ub=b, method='highs')

# Création de la grille pour les courbes de niveau
x = np.linspace(0, 5, 400)
y = np.linspace(0, 5, 400)
X, Y = np.meshgrid(x, y)

# Calcul des valeurs de la fonction de coût sur la grille
Z_initial = c_initial[0] * X + c_initial[1] * Y
Z_modified = c_modified[0] * X + c_modified[1] * Y

# Tracé des courbes de niveau et des contraintes
plt.figure(figsize=(10, 8))

# Courbe de niveau de la fonction de coût initiale à la valeur de la solution optimale
level_initial = result_initial.fun
contour_initial = plt.contour(X, Y, Z_initial, levels=[level_initial], colors='purple', linestyles='dotted')
plt.clabel(contour_initial, inline=1, fontsize=10)

# Courbe de niveau de la fonction de coût modifiée à la valeur de la solution optimale
level_modified = result_modified.fun
contour_modified = plt.contour(X, Y, Z_modified, levels=[level_modified], colors='red', linestyles='dotted')
plt.clabel(contour_modified, inline=1, fontsize=10)

# Tracé des contraintes
for i in range(len(A)):
    plt.plot(x, (b[i] - A[i, 0] * x) / A[i, 1], label=f'Constraint {i+1}')

# Tracé des solutions optimales
plt.plot(result_initial.x[0], result_initial.x[1], 'mo', label='Initial optimal solution')
plt.plot(result_modified.x[0], result_modified.x[1], 'ro', label='Modified optimal solution')

plt.xlim((0, 5))
plt.ylim((0, 5))
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Small changes are important')
plt.grid(True)
plt.show()
