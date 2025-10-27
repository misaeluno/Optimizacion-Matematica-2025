from mpl_toolkits.mplot3d import Axes3D  # Necesario para proyecciones 3D
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import numpy as np
import random
import time
# encontrar la funcion
# f(x,y) = (x^2 + y −11)^2 +(x  + y^2 −7)^2


#funcion inicial respecto a X["x","y"]
def f(x):
    return ((x[0]**2 + x[1] -11)**2 +(x[0] + x[1]**2 -7)**2)
#--------------------------------------------

#funcion grandiente de la funcion inicial
def gradiente_f(x):
    #el comando "np.array" sirve para difinir que el valor sera un vector
    #separado por la ","
    dx = ((2*(x[0]**2 + x[1] -11)*(2*x[0])) + 2*(x[0] + x[1]**2 -7))
    dy = (2*(x[0]**2 + x[1] -11)) + (2*(x[0] + x[1]**2 -7)*(2*x[1]))
    return np.array([dx,dy])
#--------------------------------------------

#similimar a la matriz hessiana inversa
def Pk(Hk, gradiente):
    # Se usa la multiplicación matricial @
    pk = -Hk @ gradiente
    return pk
#--------------------------------------------

#Calcular alpha nuevo 
def alfa(pk, x, gradiente, funcion_ini):
    c = 0.0001
    iteraciones_alpha = 100                  #valor para cambiar alpha si es que esta nunca es suficnetemente menor
    alpha = 1.0
    Pk_grad = np.dot(gradiente, pk)          #para el PC es mas facil hacer esta operacion por separado
    
    for _ in range(iteraciones_alpha):
        n = x + (alpha * pk)                 #se necesita un valor tipo vector para usar la funcion
        auxiliar = f(n)
        #mientras que:
            #La funcion inicial segun el vector N   sea mayor a     
                #funcion inicial original + c * alpha y * el gradiente inversa * Pk 
        if auxiliar <= funcion_ini + (c * alpha * Pk_grad):
            break                            # Condición cumplida
        alpha *= 0.5                         # Reducir el paso  
    
    return alpha
#--------------------------------------------

#calcular nuevo X
def x1(x,alpha,pk):
    x1= x + (alpha*pk)
    return(x1)
#--------------------------------------------

#calculkalr H nuevo
def Hk1(x, xk1, gradiente_f, H):
    Sk = xk1 -x                             #calcular Sk= X_nuevo - X_original
    Yk = gradiente_f(xk1) - gradiente_f(x)  #Calcular Yk= Gradeinte de X_nuevo - Gradiente de X_original
    I = np.identity(len(x))                 #I va hacer la matriz identidad
    # Evitar división por cero
    Yk_Sk = np.dot(Yk, Sk)                  #Producto punto entre Yk * Sk
    if abs(Yk_Sk) < 1e-10:                  #si este resultado es muy pequeño conseguimos 
        return H                            #el K nuevo
    
    # Fórmula BFGS
    term1 = I - np.outer(Sk, Yk) /Yk_Sk
    term2 = I - np.outer(Yk, Sk) /Yk_Sk
    term3 = np.outer(Sk, Sk) /Yk_Sk
    
    Hk_1 = term1 @ H @ term2 + term3
    
    return Hk_1
#--------------------------------------------

#main
def BFGS():
    #el H iniciial es la matriz identidad
    H = np.identity(2)
    #el valor inicial de X es el punto -10 y +10
    x = np.array([4,-2])
    max_iteraciones = 1000
    tolerancia = 1e-6
    cont=0
    # Almacenar trayectoria
    trayectoria = [x.copy()]

    for i in range(max_iteraciones):
        cont=cont+1 
        # Calcular función y gradiente
        funcion = f(x)
        gradiente = gradiente_f(x)

        # Verificar convergencia
        norma_grad = np.linalg.norm(gradiente)
        if norma_grad < tolerancia:
            print(f"Convergencia alcanzada en iteración {i}")
            break
        
        #calular cosas nuevas
        pk = Pk(H, gradiente)                       # Calcular dirección de búsqueda
        alpha = alfa(pk, x, gradiente, funcion)     # Búsqueda de línea
        X = x1(x, alpha, pk)                        # Actualizar x
        H = Hk1(x, X, gradiente_f, H)               # Actualizar Hk
        #el X original sera el valor de X nuevo 
        x = X                               
        trayectoria.append(x.copy())
            
    return x, trayectoria

  
# Ejecutar optimización
x_optimo, trayectoria = BFGS()
print("\n" + "="*50)
print("RESULTADO FINAL")
print("="*50)
print(f"Punto óptimo encontrado: x = {x_optimo}")
print(f"Valor de la función: f(x) = {f(x_optimo):.10f}")
print("el valor de lafuncion teorica es 0.0")
print(f"Gradiente final: ∇f(x) = {gradiente_f(x_optimo)}")
print(f"Número de iteraciones: {len(trayectoria) - 1}")

# gracios de marco de graficos
fig = plt.figure(figsize=(14, 6))

# Datos necesarios para los gráficos
x_range = np.linspace(-6, 6, 100)
y_range = np.linspace(-6, 6, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (X**2 + Y - 11)**2 + (X + Y**2 - 7)**2  # Función de Himmelblau

# Trazar trayectoria
trayectoria_array = np.array(trayectoria)

# Grafico 1
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

z_tray = [f(punto) for punto in trayectoria]
ax1.plot(trayectoria_array[:, 0], trayectoria_array[:, 1], z_tray, 
         'r.-', linewidth=2, markersize=8, label='Trayectoria BFGS')

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Optimización BFGS - Vista 3D')
ax1.legend()

# Gráfico de contorno
ax2 = fig.add_subplot(122)

contour = ax2.contour(X, Y, Z, levels=30, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.plot(trayectoria_array[:, 0], trayectoria_array[:, 1], 
         'r.-', linewidth=2, markersize=8, label='Trayectoria BFGS')

ax2.scatter([4], [-2], color='blue', s=100, marker='o', 
            label='Punto inicial', zorder=5)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Optimización BFGS - Curvas de nivel')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()