from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def f_objetivo(x):
    
    return ( x[0]-2)**2 + (x[1]+3)**2

def g_objetivo(x):
    dx = 2* (x[0]-2)
    dy = 2* (x[1]+3)
    return(np.array([dx,dy]))

def dg_estocastico(f, grad_f, x_ctual, alpha, escala, max_iter=100, tol=1e-6):
    x_historial = [x_ctual]
    for _ in range(max_iter):
        gradient= grad_f(x_actual)
        ruido = np.random.normal(0,escala, gradient.shape[0])
        gradient_estocastico = gradient + ruido
        x_nuevo = x_ctual - alpha*gradient_estocastico
        x_historial.append(x_nuevo)
        x_ctual = x_nuevo
    x_historial = np.array(x_historial)
    f_optimo = f(x_nuevo)
    return x_nuevo, f_optimo, x_historial
    
alpha = 0.1
escala= 1
x_actual= np.array([-1,1])
resultado = dg_estocastico(f_objetivo, g_objetivo, x_actual, alpha, escala)
print(resultado[0])

x = np.linspace(-2,6,100)
y = np.linspace(-8,2,100)
X,Y = np.meshgrid(x,y)
Z = f_objetivo([X,Y])

plt.figure(figsize= (4,4))
plt.contour(X,Y,Z, levels= 20, cmap="viridis")
plt.plot(resultado[2][:,0],resultado[2][:,1],"ro-")
plt.plot(2,-3, "g*",markersize=10 )
plt.show()