from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import TextBox
import matplotlib.pyplot as plt
import numpy as np

# Función objetivo: f(x,y) = (x-2)^2 + (y+3)^2
def f(x):
    return (x[0] - 2)**2 + (x[1] + 3)**2

# Gradiente de la función
def F(x):
    return np.array([2*(x[0] - 2), 2*(x[1] + 3)])

# Dirección de búsqueda
def Pk(Hk, gradiente):
    pk = -Hk @ gradiente  # Producto matricial
    return pk

# Búsqueda de línea (Armijo backtracking)
def alfa(pk, x, gradiente, funcion_ini):
    c = 0.0001
    iteraciones_alpha = 100
    alpha = 1.0
    Pk_grad = np.dot(gradiente, pk)
    
    for _ in range(iteraciones_alpha):
        n = x + (alpha * pk)
        auxiliar = f(n)
        
        if auxiliar <= funcion_ini + (c * alpha * Pk_grad):
            break
        alpha *= 0.5
    
    return alpha

# Actualización de x
def x_nuevo(xk, alpha, pk):
    return xk + (alpha * pk)

# Actualización de Hk (fórmula BFGS)
def Hk_nuevo(xk, xk1, F_func, Hk):
    Sk = xk1 - xk
    Yk = F_func(xk1) - F_func(xk)
    
    # Evitar división por cero
    Yk_Sk = np.dot(Yk, Sk)
    if abs(Yk_Sk) < 1e-10:
        return Hk
    
    I = np.identity(len(xk))
    
    # Fórmula BFGS
    rho = 1.0 / Yk_Sk
    term1 = I - rho * np.outer(Sk, Yk)
    term2 = I - rho * np.outer(Yk, Sk)
    term3 = rho * np.outer(Sk, Sk)
    
    Hk1 = term1 @ Hk @ term2 + term3
    
    return Hk1

# Main - Método BFGS
def optimizar_bfgs():
    # Inicialización
    Hk = np.identity(2)
    x = np.array([-10.0, 10.0])
    max_iteraciones = 1000
    tolerancia = 1e-6
    
    # Almacenar trayectoria
    trayectoria = [x.copy()]
    
    for i in range(max_iteraciones):
        # Calcular función y gradiente
        funcion = f(x)
        gradiente = F(x)
        
        # Verificar convergencia
        norma_grad = np.linalg.norm(gradiente)
        if norma_grad < tolerancia:
            print(f"Convergencia alcanzada en iteración {i}")
            break
        
        # Calcular dirección de búsqueda
        pk = Pk(Hk, gradiente)
        
        # Búsqueda de línea
        alpha = alfa(pk, x, gradiente, funcion)
        
        # Actualizar x
        x_next = x_nuevo(x, alpha, pk)
        
        # Actualizar Hk
        Hk = Hk_nuevo(x, x_next, F, Hk)
        
        # Actualizar x para la siguiente iteración
        x = x_next
        trayectoria.append(x.copy())
        
        # Imprimir progreso cada 10 iteraciones
        if i % 10 == 0:
            print(f"Iteración {i}: x = {x}, f(x) = {funcion:.6f}")
    
    return x, trayectoria

# Ejecutar optimización
x_optimo, trayectoria = optimizar_bfgs()

print("\n" + "="*50)
print("RESULTADO FINAL")
print("="*50)
print(f"Punto óptimo encontrado: x = {x_optimo}")
print(f"Valor de la función: f(x) = {f(x_optimo):.10f}")
print(f"Gradiente final: ∇f(x) = {F(x_optimo)}")
print(f"Solución teórica: x = [2, -3]")
print(f"Número de iteraciones: {len(trayectoria) - 1}")

# Visualización 3D
fig = plt.figure(figsize=(14, 6))

# Gráfico 3D
ax1 = fig.add_subplot(121, projection='3d')
x_range = np.linspace(-12, 12, 100)
y_range = np.linspace(-8, 12, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (X - 2)**2 + (Y + 3)**2

ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')

# Trazar trayectoria
trayectoria_array = np.array(trayectoria)
z_tray = [f(punto) for punto in trayectoria]
ax1.plot(trayectoria_array[:, 0], trayectoria_array[:, 1], z_tray, 
         'r.-', linewidth=2, markersize=8, label='Trayectoria BFGS')
ax1.scatter([2], [-3], [0], color='green', s=200, marker='*', 
            label='Óptimo teórico', zorder=5)

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
ax2.scatter([2], [-3], color='green', s=200, marker='*', 
            label='Óptimo teórico', zorder=5)
ax2.scatter([-10], [10], color='blue', s=100, marker='o', 
            label='Punto inicial', zorder=5)

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Optimización BFGS - Curvas de nivel')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()