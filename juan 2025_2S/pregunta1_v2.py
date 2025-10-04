# Pregunta 1

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Pregunta 1")
print("")

print("Encontrar el mínimo de la función:")
print("f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2")
print("")

print("Método GD")
print("")

# Función objetivo
def f_multi(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Gradiente de la función
def grad_f_multi(x):
    dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([dx, dy])

# Algoritmo de descenso de gradiente
def dg_multi(x_inicial, alpha, max_iter = 10000, tol = 1e-6):
    resultados = []
    for k in range(len(alpha)):
        x_actual = x_inicial.copy()
        for i in range(max_iter):
            x_nuevo = x_actual- alpha[k] * grad_f_multi(x_actual)
            criterio_1 = np.linalg.norm(grad_f_multi(x_nuevo))
            criterio_2 = np.linalg.norm(x_nuevo- x_actual)
            if (criterio_1 < tol or criterio_2 < tol):
                resultados.append({'Alpha': alpha[k],
                                   'x': x_nuevo[0],
                                   'y': x_nuevo[1],
                                   'f(x, y)': f_multi(x_nuevo),
                                   'Iteraciones': i + 1})
                break
            x_actual = x_nuevo
        else:
            resultados.append({'Alpha': alpha[k],
                                'x': x_actual[0],
                               'y': x_actual[1],
                               'f(x, y)': f_multi(x_actual),
                               'Iteraciones': max_iter})
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

# Ajuste de parámetros del algoritmo
alpha = np.array([0.001, 0.005, 0.01])

# Definición del punto inicial
x_inicial = np.array([-2.5, 2.5])

# Ejecución del algoritmo y resultados
r_gd = dg_multi(x_inicial, alpha)
print(r_gd)
print("")

print("Método GDM (usar alpha[1] = 0.005)")
print("")

# Algoritmo de descenso de gradiente con momento
def dgm_multi(x_inicial, gamma, alpha_value, max_iter = 10000, tol = 1e-6):
    resultados = []
    for k in range(len(gamma)):
        x_actual = x_inicial.copy()
        velocidad = np.zeros(2)
        for i in range(max_iter):
            velocidad = gamma[k] * velocidad + alpha_value * grad_f_multi(x_actual)
            x_nuevo = x_actual- velocidad
            criterio_1 = np.linalg.norm(grad_f_multi(x_nuevo))
            criterio_2 = np.linalg.norm(x_nuevo- x_actual)
            if (criterio_1 < tol or criterio_2 < tol):
                resultados.append({'Gamma': gamma[k],
                                       'x': x_nuevo[0],
                                       'y': x_nuevo[1],
                                       'f(x, y)': f_multi(x_nuevo),
                                       'Iteraciones': i + 1})
                break
            x_actual = x_nuevo
        else:
            resultados.append({'Gamma': gamma[k],
                               'x': x_actual[0],
                                'y': x_actual[1],
                               'f(x, y)': f_multi(x_actual),
                               'Iteraciones': max_iter})
    df_resultados = pd.DataFrame(resultados)
    return df_resultados

# Parámetros del algoritmo
x_inicial = np.array([-2.5,2.5])

# Tasa de aprendizaje
alpha_gdm = alpha[1]

# Coeficiente de momento
gamma = np.array([0.5, 0.7, 0.9])

# Ejecución del algoritmo y resultados
r_gdm = dgm_multi(x_inicial, gamma, alpha_gdm)
print(r_gdm)
