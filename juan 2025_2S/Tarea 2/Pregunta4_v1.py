# Pregunta 4

import numpy as np
import matplotlib.pyplot as plt

print("""Pregunta 4

f(x, y) = 0.5x^2 + 2.5y^2 - 2xy - x^3
""")

# Función objetivo
def f_multi(x):
    return 0.5*x[0]**2 + 2.5*x[1]**2 - 2*x[0]*x[1] - x[0]**3

# Gradiente de la función
def grad_f_multi(x):
    dx = x[0] - 2*x[1] - 3*x[0]**2
    dy = 5*x[1] - 2*x[0]
    return np.array([dx, dy])

# Hessiana
def hess_f_multi(x):
    dxx = 1 - 6*x[0]
    dxy = - 2
    dyy = 5
    return np.array([[dxx, dxy], [dxy, dyy]])

x0 = np.array([1.5, 0.5])
H = hess_f_multi(x0)
eigenvalues = np.linalg.eigvals(H)

print(f"\nHessiana en x0 = (1.5, 0.5):")
print(H)
print(f"\nValores propios (eigenvalues):")
print(eigenvalues)
print(f"\nMínimo valor propio: {np.min(eigenvalues):.6f}")

if np.all(eigenvalues > 0):
    print("La Hessiana ES positiva definida (todos los eigenvalues > 0)")
else:
    print("La Hessiana NO ES positiva definida (hay eigenvalues ≤ 0)")
    print("  Esto puede causar problemas en el método de Newton estándar")

print("-"*70)
print("Newton estándar")
print()

def newton_estandar(x_actual, max_iter=10000, tol=1e-6):
    x_historial = np.array([x_actual])
    try:
        for i in range(max_iter):
            H = hess_f_multi(x_actual)
            g = grad_f_multi(x_actual)
            
            # Intentar invertir la Hessiana sin modificar
            x_nuevo = x_actual - np.linalg.inv(H) @ g
            x_historial = np.vstack((x_historial, x_nuevo))
            
            criterio_1 = np.linalg.norm(g)
            criterio_2 = np.linalg.norm(x_nuevo - x_actual)
            
            if (criterio_1 < tol or criterio_2 < tol):
                break
            x_actual = x_nuevo
            
        return x_nuevo, x_historial, True
    except:
        return x_actual, x_historial, False

x0 = np.array([1.5, 0.5])
r_estandar, hist_estandar, exito = newton_estandar(x0)

print(f"Punto inicial: {x0}")
if exito:
    print(f"Punto final: ({r_estandar[0]}, {r_estandar[1]})")
    print(f"f(x, y) = {f_multi(r_estandar)}")
    print(f"Iteraciones: {len(hist_estandar)}")
else:
    print("El algoritmo falló (no pudo invertir la Hessiana)")

print("-"*70)
print("Newton regularizado")
print()

def newton_regularizado(x_actual, max_iter=10000, tol=1e-6):
    x_historial = np.array([x_actual])
    
    for i in range(max_iter):
        H = hess_f_multi(x_actual)
        g = grad_f_multi(x_actual)
        
        # Verificar eigenvalues
        eigenvalues = np.linalg.eigvals(H)
        min_eig = np.min(eigenvalues)
        
        # Regularización adaptativa
        if min_eig <= 1e-6:  # Umbral pequeño positivo
            c = -min_eig + 0.1  # Forzar que sea positiva definida
            H_modificada = H + c * np.eye(2)
            if i < 10:
                print(f"  Iteración {i}: Regularizando con c = {c:.4f}")
        else:
            H_modificada = H
        
        # Paso de Newton
        x_nuevo = x_actual - np.linalg.inv(H_modificada) @ g
        x_historial = np.vstack((x_historial, x_nuevo))
        
        # Convergencia
        if np.linalg.norm(g) < tol or np.linalg.norm(x_nuevo - x_actual) < tol:
            break
        
        x_actual = x_nuevo
    
    return x_nuevo, x_historial

x0 = np.array([1.5, 0.5])
r_regularizado, hist_regularizado = newton_regularizado(x0)

print(f"\nPunto inicial: {x0}")
print(f"Punto final: ({r_regularizado[0]}, {r_regularizado[1]})")
print(f"f(x, y) = {f_multi(r_regularizado)}")
print(f"Iteraciones: {len(hist_regularizado)}")

# Crear gráfico de contorno
x_range = np.linspace(-1, 3, 400)
y_range = np.linspace(-1, 2, 400)
X, Y = np.meshgrid(x_range, y_range)
Z = f_multi([X, Y])

plt.figure(figsize=(12, 5))

# Gráfico 1: Método Estándar
plt.subplot(1, 2, 1)
contour = plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.clabel(contour, inline=1, fontsize=8)

if exito and len(hist_estandar) > 1:
    plt.plot(hist_estandar[:, 0], hist_estandar[:, 1], 
             'ro-', linewidth=2, markersize=6, label='Trayectoria')
    plt.plot(hist_estandar[0, 0], hist_estandar[0, 1], 
             'gs', markersize=10, label='Inicio')
    plt.plot(hist_estandar[-1, 0], hist_estandar[-1, 1], 
             'r*', markersize=15, label='Final')
else:
    plt.plot(x0[0], x0[1], 'gs', markersize=10, label='Inicio')

plt.title('Método de Newton Estándar', fontsize=12, fontweight='bold')
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

# Gráfico 2: Método Regularizado
plt.subplot(1, 2, 2)
contour = plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.clabel(contour, inline=1, fontsize=8)

plt.plot(hist_regularizado[:, 0], hist_regularizado[:, 1], 
         'bo-', linewidth=2, markersize=6, label='Trayectoria')
plt.plot(hist_regularizado[0, 0], hist_regularizado[0, 1], 
         'gs', markersize=10, label='Inicio')
plt.plot(hist_regularizado[-1, 0], hist_regularizado[-1, 1], 
         'b*', markersize=15, label='Final')

plt.title('Método de Newton Regularizado', fontsize=12, fontweight='bold')
plt.xlabel('x', fontsize=10)
plt.ylabel('y', fontsize=10)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

