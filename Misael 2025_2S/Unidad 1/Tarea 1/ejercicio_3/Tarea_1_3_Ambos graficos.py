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

#------------------------------------------------
#DATOS ORIGINALES
# Freceuncia Reloj
X = np.array([4.29, 4.34, 2.72, 4.08, 3.60,
3.30, 3.84, 2.34, 3.64, 3.76,
3.14, 3.80, 4.34, 2.64, 3.16,
4.35, 4.45, 2.29, 3.19, 3.40,
4.26, 2.35, 4.47, 4.37, 2.21])

#consumo de Energia
Y = np.array([122, 130, 88, 121, 108,
102, 107, 75, 105, 119,
102, 112, 125, 79, 98,
127, 121, 73, 105, 101,
117, 78, 123, 119, 70])
#------------------------------------------------

#DATOS EXTRAS
# Freceuncia Reloj
x = np.array([4.29, 4.34, 2.72, 4.08, 3.60,
3.30, 3.84, 2.34, 3.64, 3.76,
3.14, 3.80, 4.34, 2.64, 3.16,
4.35, 4.45, 2.29, 3.19, 3.40,
4.26, 2.35, 4.47, 4.37, 2.21, 2.5,4.5])

#consumo de Energia
y = np.array([122, 130, 88, 121, 108,
102, 107, 75, 105, 119,
102, 112, 125, 79, 98,
127, 121, 73, 105, 101,
117, 78, 123, 119, 70, 130,70])
#------------------------------------------------

#cantidad de filas en la matriz ORIGINAL
N=len(X)
Beta0 = 1
Beta1 = 1
iteraciones=50
resto=0
alpha = 0.02
#------------------------------------------------

#cantidad de filas en la matriz EXTRAS
n=len(x)
Teta0 = 1
Teta1 = 1
contador=50
restante=0
alfa = 0.02
#--------------------------------------------------

# Lista  ORIGINAL para guardar el valor de costo J en cada iteración
costos = []
#--------------------------------------------------

# Lista  EXTRAS para guardar el valor de costo J en cada iteración
deudas = []
#--------------------------------------------------

# Crear mallas ORIGINAL para Beta0 y Beta1
Beta0_malla = np.linspace(-2, 2, 100)
Beta1_malla = np.linspace(-2, 2, 100)
T0, T1 = np.meshgrid(Beta0_malla, Beta1_malla)
#--------------------------------------------------

# Crear mallas EXTRAS para Beta0 y Beta1
Teta0_malla = np.linspace(-2, 2, 100)
Teta1_malla = np.linspace(-2, 2, 100)
t0, t1 = np.meshgrid(Teta0_malla, Teta1_malla)

# Variable para guardar la línea actual ORIGINAL y poder borrarla
linea_actual = None
#--------------------------------------------------

# Variable para guardar la línea actual EXTRA y poder borrarla
linea_ahora = None
#--------------------------------------------------

# Calcular J(teta_0, teta_1) para cada combinación ORIGINAL
Z = np.zeros(T0.shape)
for i in range(T0.shape[0]):
    for j in range(T0.shape[1]):
        errores = hteta(T0[i, j], T1[i, j], X) - Y
        Z[i, j] = (1 / (2 * N)) * np.sum(errores ** 2)
#--------------------------------------------------

# Calcular J(teta_0, teta_1) para cada combinación EXTRA
z = np.zeros(t0.shape)
for i in range(t0.shape[0]):
    for j in range(t0.shape[1]):
        errores = hteta(t0[i, j], t1[i, j], x) - y
        z[i, j] = (1 / (2 * n)) * np.sum(errores ** 2)

# Opcional: dibujar trayectoria del descenso de gradiente ORIGINAL
Beta0_hist = []
Beta1_hist = []
#--------------------------------------------------

# Opcional: dibujar trayectoria del descenso de gradiente EXTRAS
Teta0_hist = []
Teta1_hist = []

# Descenso de gradiente (100 iteraciones) con J(teta0: teta1)
for i in range(iteraciones):
#para gradeinte, costo y curva de nivel
    error = sumatori(Beta0, Beta1, Y, X)    #ORIGINAL
    problema = sumatori(Teta0, Teta1, y, x) #EXTRAS

    #Nuevo costo para ORIGINAL
    costo = (1 / (2 * N)) * np.sum(error ** 2)
    costos.append(costo)
    #-------------------------------------------

    #Nuevo costo para EXTRAS
    deuda = (1 / (2 * n)) * np.sum(problema ** 2)
    deudas.append(deuda)
    #--------------------------------------------
    
    #Calcular nuebos Betas para ORIGINAL
    Beta0 -= alpha * (1/N) * np.sum(error)
    Beta1 -= alpha * (1/N) * np.sum(error * X)
    #---------------------------------------------

    #Calcular nuebos Tetas para EXTRAS
    Teta0 -= alfa * (1/n) * np.sum(problema)
    Teta1 -= alfa * (1/n) * np.sum(problema * x)
#------------------------

#para curva de nivel
    Beta0_hist.append(Beta0)
    Beta1_hist.append(Beta1)
#------------------------

#para curva de nivel
    Teta0_hist.append(Teta0)
    Teta1_hist.append(Teta1)
#------------------------
# Crear la segunda figura
fig, ax5 = plt.subplots()

#Buscar minimos y maximos de ORIGINAL
minoX = X.min(axis=0)
minoY = Y.min(axis=0)
maxiX = X.max(axis=0)
maxiY = Y.max(axis=0)
#----------------------------------------

#Buscar minimos y maximos de EXTRAS
minimoX = x.min(axis=0)
minimoY = y.min(axis=0)
maximoX = x.max(axis=0)
maximoY = y.max(axis=0)
#----------------------------------------

#puntos y graicos de ORIGINAL
ax5.scatter(X, Y, c="red", label='Datos originales', marker="x")
linea_actual, = ax5.plot(X,(Beta1*X)+Beta0 , color="blue", label="pendiente ORIGINAL")
#----------------------------------------

#Puntos y graficos de EXTRAS
ax5.scatter([2.5,4.5], [130,70], c="orange", label='Datos extras', marker="x")
linea_ahora, = ax5.plot(x,(Teta1*x)+Teta0 , color="purple", label="pendiente EXTRAS")
#----------------------------------------

#Leyendas de ORIGINAL
ax5.set_title('Grafico con valor original')
ax5.set_xlabel('ciclo de reloj')
ax5.set_ylabel('energia')
ax5.legend()
#--------------------------
#       Emprimir graficos
#--------------------------
plt.show()