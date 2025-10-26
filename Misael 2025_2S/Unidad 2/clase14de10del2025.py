from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def f_multivariables(x):
    aux1 = 3 * (1-x[0]) ** 2 * np.exp(-x[0] ** 2 - (x[1] +1) **2)
    aux2 = -10* (x[0] / 5 -x[0]**3 - x[1] **5) * np.exp(-x[0] ** 2 - x[1] **2)
    aux3 = -(1 / 3 ) * np.exp(-(x[0] + 1) ** 2 -x[1] **2)
    return aux1 + aux2 + aux3

def simulated_annealing(f, x_actual, num_iter, tamano_paso = 2, t_actual = 100, alpha = 0.95):
    x_historial = [x_actual]
    f_actual = f(x_actual)
    x_mejor = x_actual
    f_mejor = f_actual
    
    for _ in range(num_iter):
        #vecinos alatorios
        if isinstance(x_actual, np.ndarray):
            dx = np.random.normal(0, tamano_paso, x_actual.shape[0])
        else :
            dx = np.random.normal(0, tamano_paso)
        x_nuevo = x_actual + dx
        f_nuevo = f(x_nuevo)
        x_historial.append(x_nuevo)
        #delta de la funcion objetivo
        delta = f_nuevo - f_actual

        if delta >0 :
            x_actual = x_nuevo
            f_actual = f_nuevo
            if f_actual > f_mejor:
                x_mejor = x_actual
                f_mejor = f_actual
        else :
            proba_aceptacion = np.exp(delta / t_actual)
            if np.random.rand() < proba_aceptacion:
                x_actual = x_nuevo
                f_actual = f_actual
        t_actual = alpha - t_actual
    x_historial = np.array(x_historial)
    return x_mejor, f_mejor, x_historial


x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
X, Y = np.meshgrid(x,y)
Z = f_multivariables([X,Y])

fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111, projection = "3d")
ax.plot_surface(X, Y, Z, cmap = "viridis", alpha = 0.9)
ax.view_init(elev = 15, azim = 45)
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.tick_params(labelsize = 8)
plt.show()

np.random.seed(2025)
x_actual = np.array([-2, -2 ])
resultado = simulated_annealing(f_multivariables, x_actual, num_iter=10000)
print("x = ", resultado[0])
print("f(x,y) = ", resultado[1])
