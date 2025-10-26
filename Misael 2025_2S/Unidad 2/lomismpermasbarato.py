from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def f(x):
    aux1 = np.exp(-x**2 / 50) * np.sin(2*x)
    aux2 = 0.5*np.exp(-(x-5)**2 /10) * np.cos(3*x)
    aux3 = 0.3*np.exp(-(x+5)**2 / 8) * np.sin(4*x)
    return(aux1+aux2+aux3)

def comprobacion(x_actual, epsilon):
    
    if (x_actual + epsilon)< -20 :
        epsilon = np.random.normal(0,tamano_paso)
        return comprobacion(x_actual,epsilon)

    elif (x_actual + epsilon)>20 :
        epsilon = np.random.normal(0,tamano_paso)
        return comprobacion(x_actual,epsilon)
    
    else:
        x_nuevo = x_actual + epsilon
        return x_nuevo


numero_iteraciones= 10000
x_actual= -10
f_actual = f(x_actual)
tamano_paso=5
t_actual = 100
alpha = 0.95
historia_puntos=[np.array([x_actual,f_actual])]

x_mejor = x_actual
f_mejor = f_actual

for _ in range(numero_iteraciones):
    epsilon = np.random.normal(0,tamano_paso)

    x_nuevo = comprobacion(x_actual,epsilon)
    f_nuevo=f(x_nuevo)
    
    historia_puntos.append(np.array([x_nuevo,f_nuevo]))

    delta = f_nuevo - f_actual

    if delta > 0 :
        x_actual = x_nuevo
        f_actual = f_nuevo
        if f_actual > f_mejor :
            x_mejor = x_actual
            f_mejor = f_actual
    else :
        prob_aceptacion = np.exp(delta/t_actual)
        if np.random.rand()< prob_aceptacion:
            x_actual = x_nuevo
            f_actual = f_nuevo
    t_actual = alpha * t_actual

historia_puntos = np.array(historia_puntos)



print(x_mejor, f_mejor)




x = np.linspace(-20,20,1000)
y = f(x)
plt.figure(figsize=(10,3))
plt.plot(x,y)
plt.plot(historia_puntos[:,0], historia_puntos[:,1], "ro" )
plt.plot(historia_puntos[0,0], historia_puntos[0,1], "b*" )
plt.plot(x_mejor, f_mejor, "g*", markersize=15)
plt.show()