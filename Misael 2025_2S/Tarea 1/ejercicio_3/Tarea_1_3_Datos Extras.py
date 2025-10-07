from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
import matplotlib.pyplot as plt
import numpy as np
import random
import time

def hteta(beta0: float, beta1: float, x: np.array):
     teta0=beta0
     teta1=beta1
     X=x
     resultado = (teta1*X)+teta0
     return resultado

def sumatori(beta0: float, beta1: float, y: np.array, x: np.array):
    teta0 = beta0
    teta1 = beta1
    Y = y
    X = x
    resultado = (hteta(teta0,teta1,X) - Y)
    return resultado

# Freceuncia Reloj
X = np.array([4.29, 4.34, 2.72, 4.08, 3.60,
3.30, 3.84, 2.34, 3.64, 3.76,
3.14, 3.80, 4.34, 2.64, 3.16,
4.35, 4.45, 2.29, 3.19, 3.40,
4.26, 2.35, 4.47, 4.37, 2.21, 2.5,4.5])

#consumo de Energia
Y = np.array([122, 130, 88, 121, 108,
102, 107, 75, 105, 119,
102, 112, 125, 79, 98,
127, 121, 73, 105, 101,
117, 78, 123, 119, 70, 130,70])

#print(Y)

#cantidad de filas en la matriz
N=len(X)
print(N)
Beta0 = 1
Beta1 = 1
iteraciones=50
resto=0
alpha = 0.02

# Lista para guardar el valor de costo J en cada iteración
costos = []

# Crear mallas para theta0 y theta1
Beta0_malla = np.linspace(-2, 2, 100)
Beta1_malla = np.linspace(-2, 2, 100)
T0, T1 = np.meshgrid(Beta0_malla, Beta1_malla)

# Variable para guardar la línea actual y poder borrarla
linea_actual = None

# Calcular J(teta_0, teta_1) para cada combinación
Z = np.zeros(T0.shape)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        errores = hteta(T0[i, j], T1[i, j], X) - Y
        Z[i, j] = (1 / (2 * N)) * np.sum(errores ** 2)

# Opcional: dibujar trayectoria del descenso de gradiente
Beta0_hist = []
Beta1_hist = []

# Descenso de gradiente (100 iteraciones) con J(teta0: teta1)
for i in range(iteraciones):
#para gradeinte, costo y curva de nivel
    error = sumatori(Beta0, Beta1, Y, X)
    costo = (1 / (2 * N)) * np.sum(error ** 2)
    costos.append(costo)

    Beta0 -= alpha * (1/N) * np.sum(error)
    Beta1 -= alpha * (1/N) * np.sum(error * X)
#------------------------

#para curva de nivel
    Beta0_hist.append(Beta0)
    Beta1_hist.append(Beta1)
#------------------------

fig, ax5 = plt.subplots()

minoX = X.min(axis=0)
minoY = Y.min(axis=0)
maxiX = X.max(axis=0)
maxiY = Y.max(axis=0)

ax5.scatter(X, Y, c="red", label='Datos originales', marker="x")
ax5.scatter([2.5,4.5], [130,70], c="orange", label='Datos extras', marker="x")
linea_actual, = ax5.plot(X,(Beta1*X)+Beta0 , color="blue")

#print((X*Beta1)*Beta0)
#print(X)
ax5.set_title('grafico con valor extra')
ax5.set_xlabel('ciclo de reloj')
ax5.set_ylabel('energia')
ax5.legend()

#--------------------------
#       Emprimir graficos
#--------------------------
plt.show()