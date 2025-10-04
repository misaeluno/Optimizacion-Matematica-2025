# Pregunta 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("Pregunta 2")
print("")

print("Minimizar la función de Rosenbrock:")
print("f(x, y) = (1 - x)^2 + 100*(y - x^2)^2")
print("")

# Función objetivo
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Gradiente de la función
def grad_rosenbrock(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# Decaimiento por tiempo: alpha_k = alpha_0/(1 + k*tau)
def dg_tiempo(x_actual, alpha_0, tau, max_iter, tol):
    x_historial = [x_actual.copy()]
    alpha_historial = [alpha_0]
    
    for k in range(max_iter):
        alpha_k = alpha_0/(1 + k*tau)
        alpha_historial.append(alpha_k)
        
        x_nuevo = x_actual - alpha_k*grad_rosenbrock(x_actual)
        x_historial.append(x_nuevo.copy())
        
        criterio_1 = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_2 = np.linalg.norm(x_nuevo - x_actual)
        
        if criterio_1 < tol or criterio_2 < tol:
            break
        
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(alpha_historial)


# Decaimiento por pasos: alpha_k = alpha_0 * d^(k/s)
def dg_pasos(x_actual, alpha_0, d, s, max_iter, tol):
    x_historial = [x_actual.copy()]
    alpha_historial = [alpha_0]
    
    for k in range(max_iter):
        alpha_k = alpha_0*(d**(k/s))
        alpha_historial.append(alpha_k)
        
        x_nuevo = x_actual - alpha_k*grad_rosenbrock(x_actual)
        x_historial.append(x_nuevo.copy())
        
        criterio_1 = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_2 = np.linalg.norm(x_nuevo - x_actual)
        
        if criterio_1 < tol or criterio_2 < tol:
            break
        
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(alpha_historial)


# Decaimiento exponencial: alpha_k = alpha_0*e^(-k*tau)
def dg_exponencial(x_actual, alpha_0, tau, max_iter, tol):
    x_historial = [x_actual.copy()]
    alpha_historial = [alpha_0]
    
    for k in range(max_iter):
        alpha_k = alpha_0*np.exp(-k*tau)
        alpha_historial.append(alpha_k)
        
        x_nuevo = x_actual - alpha_k*grad_rosenbrock(x_actual)
        x_historial.append(x_nuevo.copy())
        
        criterio_1 = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_2 = np.linalg.norm(x_nuevo - x_actual)
        
        if criterio_1 < tol or criterio_2 < tol:
            break
        
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(alpha_historial)

# Parámetros iniciales
x_inicial = np.array([-1, -1])
alpha_0 = 0.002
max_iter = 10000
tol = 1e-5 # En 1e-6, el decaimiento por pasos llegaba al límite de iteraciones

# Parámetros de decaimiento
tau = alpha_0/max_iter
d = 0.5
s = 2000

print("DECAIMIENTO POR TIEMPO:")
r_tiempo = dg_tiempo(x_inicial.copy(), alpha_0, tau, max_iter, tol)
print(f"Resultado: (x, y) = ({r_tiempo[0][0]:.3f}, {r_tiempo[0][1]:.3f})")
print(f"f(x, y) = {rosenbrock(r_tiempo[0]):.2e}")
print(f"Iteraciones: {len(r_tiempo[1]) - 1}")
print("")

print("DECAIMIENTO POR PASOS:")
r_pasos = dg_pasos(x_inicial.copy(), alpha_0, d, s, max_iter, tol)
print(f"Resultado: (x, y) = ({r_pasos[0][0]:.3f}, {r_pasos[0][1]:.3f})")
print(f"f(x, y) = {rosenbrock(r_pasos[0]):.2e}")
print(f"Iteraciones: {len(r_pasos[1]) - 1}")
print("")

print("DECAIMIENTO EXPONENCIAL:")
r_exponencial = dg_exponencial(x_inicial.copy(), alpha_0, tau, max_iter, tol)
print(f"Resultado: (x, y) = ({r_exponencial[0][0]:.3f}, {r_exponencial[0][1]:.3f})")
print(f"f(x, y) = {rosenbrock(r_exponencial[0]):.2e}")
print(f"Iteraciones: {len(r_exponencial[1]) - 1}")
print("")

