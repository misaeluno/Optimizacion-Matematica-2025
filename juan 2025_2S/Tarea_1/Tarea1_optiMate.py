import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pregunta 1
print("Pregunta 1 - Función de Himmelblau")
print()

print("Encontrar el mínimo de la función:")
print("f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2")
print("Punto inicial: (-2.5, 2.5)")
print()

# Funcion objetivo
def f_multi(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

# Gradiente de la funcion
def grad_f_multi(x):
    dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
    dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
    return np.array([dx, dy])

# Algoritmo de descenso de gradiente (GD)
def dg_multi(x_inicial, alpha, max_iter = 10000, tol = 1e-6):
    resultados = []
    trayectorias = []
    for k in range(len(alpha)):
        x_actual = x_inicial.copy()
        x_historial = np.array([x_actual])
        for i in range(max_iter):
            x_nuevo = x_actual - alpha[k]*grad_f_multi(x_actual)
            x_historial = np.vstack((x_historial, x_nuevo))
            criterio_1 = np.linalg.norm(grad_f_multi(x_nuevo))
            criterio_2 = np.linalg.norm(x_nuevo - x_actual)
            if (criterio_1 < tol or criterio_2 < tol):
                resultados.append({'Algoritmo': 'GD',
                                   'Alpha': alpha[k],
                                   'Gamma': np.nan,
                                   'x': x_nuevo[0],
                                   'y': x_nuevo[1],
                                   'f(x, y)': f_multi(x_nuevo),
                                   'Iteraciones': i + 1})
                trayectorias.append(x_historial)
                break
            x_actual = x_nuevo
        else:
            resultados.append({'Algoritmo': 'GD',
                               'Alpha': alpha[k],
                               'Gamma': np.nan,
                               'x': x_actual[0],
                               'y': x_actual[1],
                               'f(x, y)': f_multi(x_actual),
                               'Iteraciones': max_iter})
            trayectorias.append(x_historial)
    
    return pd.DataFrame(resultados), trayectorias

# Algoritmo de descenso de gradiente con momento (GDM)
def dgm_multi(x_inicial, gamma, alpha_value, max_iter = 10000, tol = 1e-6):
    resultados = []
    trayectorias = []
    
    for k in range(len(gamma)):
        x_actual = x_inicial.copy()
        velocidad = np.zeros(2)
        x_historial = np.array([x_actual])
        
        for i in range(max_iter):
            velocidad = gamma[k]*velocidad + alpha_value*grad_f_multi(x_actual)
            x_nuevo = x_actual - velocidad
            x_historial = np.vstack((x_historial, x_nuevo))
            
            criterio_1 = np.linalg.norm(grad_f_multi(x_nuevo))
            criterio_2 = np.linalg.norm(x_nuevo - x_actual)
            
            if (criterio_1 < tol or criterio_2 < tol):
                resultados.append({'Algoritmo': 'GDM',
                                   'Alpha': alpha_value,
                                   'Gamma': gamma[k],
                                   'x': x_nuevo[0],
                                   'y': x_nuevo[1],
                                   'f(x, y)': f_multi(x_nuevo),
                                   'Iteraciones': i + 1})
                trayectorias.append(x_historial)
                break
            x_actual = x_nuevo
        else:
            resultados.append({'Algoritmo': 'GDM',
                               'Alpha': alpha_value,
                               'Gamma': gamma[k],
                               'x': x_actual[0],
                               'y': x_actual[1],
                               'f(x, y)': f_multi(x_actual),
                               'Iteraciones': max_iter})
            trayectorias.append(x_historial)
    
    return pd.DataFrame(resultados), trayectorias

# Ejecucion de algoritmos

# Parametros
x_inicial = np.array([-2.5, 2.5])
alpha = np.array([0.001, 0.005, 0.01])
alpha_gdm = alpha[1] # 0.005
gamma = np.array([0.5, 0.7, 0.9])

# Ejecutar GD
print("Método GD")
print()
r_gd, tray_gd = dg_multi(x_inicial, alpha)
print(r_gd)
print("-"*70)

# Ejecutar GDM
print("Método GDM")
print()
r_gdm, tray_gdm = dgm_multi(x_inicial, gamma, alpha_gdm)
print(r_gdm)
print("-"*70)

# Identificar mejores configuraciones
mejor_gd = r_gd.loc[r_gd['Iteraciones'].idxmin()]
mejor_gdm = r_gdm.loc[r_gdm['Iteraciones'].idxmin()]

# Tabla comparativa
print("Tabla comparativa")
print()
df_completo = pd.concat([r_gd, r_gdm], ignore_index=True)
print(df_completo.to_string(index=False))
print("-"*70)

# Visualizacion

# Crear malla para el gráfico de contorno
x_range = np.linspace(-5, 5, 200)
y_range = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = f_multi([X, Y])

# Indices de las mejores configuraciones
idx_mejor_gd = r_gd['Iteraciones'].idxmin()
idx_mejor_gdm = r_gdm['Iteraciones'].idxmin()

# Crear figura
plt.figure(figsize = (10, 8))
contour = plt.contour(X, Y, Z, levels = 30, cmap = 'viridis', alpha = 0.6)
plt.clabel(contour, inline = 1, fontsize = 8)

# Graficar trayectoria de mejor GD
tray_gd_mejor = tray_gd[idx_mejor_gd]
plt.plot(tray_gd_mejor[:, 0], tray_gd_mejor[:, 1], 
         'b-', linewidth = 2, label = f'GD (Alpha = {mejor_gd["Alpha"]:.3f})', alpha = 0.7)
plt.scatter(tray_gd_mejor[:, 0], tray_gd_mejor[:, 1], 
            c = 'blue', s = 20, zorder = 5)

# Graficar trayectoria de mejor GDM
tray_gdm_mejor = tray_gdm[idx_mejor_gdm]
plt.plot(tray_gdm_mejor[:, 0], tray_gdm_mejor[:, 1], 
         'r-', linewidth = 2, label = f'GDM (Gamma = {mejor_gdm["Gamma"]:.1f})', alpha = 0.7)
plt.scatter(tray_gdm_mejor[:, 0], tray_gdm_mejor[:, 1], 
            c = 'red', s = 20, zorder = 5)

# Marcar punto inicial
plt.scatter(x_inicial[0], x_inicial[1], 
            c = 'green', s = 200, marker = '*', 
            edgecolors = 'black', linewidths = 2, 
            label = 'Punto inicial', zorder = 10)

# Marcar minimos encontrados
plt.scatter(mejor_gd['x'], mejor_gd['y'], 
            c = 'blue', s = 150, marker = 'X', 
            edgecolors = 'black', linewidths = 2, zorder = 10)
plt.scatter(mejor_gdm['x'], mejor_gdm['y'], 
            c = 'red', s = 150, marker = 'X', 
            edgecolors = 'black', linewidths = 2, zorder = 10)

plt.xlabel('x', fontsize = 12)
plt.ylabel('y', fontsize = 12)
plt.title('Función de Himmelblau: Trayectorias de Convergencia\n$f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$', 
          fontsize = 14)
plt.legend(fontsize = 10, loc = 'upper right')
plt.grid(True, alpha = 0.3)
plt.tight_layout()
plt.show()

# Conclusiones
print("Conclusiones")
print()
print("1. Convergencia:")
print(f"   - GD converge en {int(mejor_gd['Iteraciones'])} iteraciones, con Alpha = {mejor_gd['Alpha']:.3f}")
print(f"   - GDM converge en {int(mejor_gdm['Iteraciones'])} iteraciones, con Gamma = {mejor_gdm['Gamma']:.1f}")

if mejor_gdm['Iteraciones'] < mejor_gd['Iteraciones']:
    mejora = ((mejor_gd['Iteraciones'] - mejor_gdm['Iteraciones'])/mejor_gd['Iteraciones'])*100
    print(f"   - GDM es {mejora:.1f}% más rápido que GD.")
    print("""
2. Efecto del momento:
   El momento ayuda significativamente en este caso porque:
   - Acelera el descenso al acumular velocidad en direcciones consistentes.
   - Suaviza oscilaciones en regiones con alta curvatura.
   - La función de Himmelblau tiene valles pronunciados donde el
     momento permite atravesarlos más rápidamente.
     """)
else:
    print("   - GD converge más rápido en este caso particular.")
    print("""
2. Efecto del momento:
   El momento NO mejora el rendimiento porque:
   - El valor de Gamma puede ser demasiado bajo o alto para este problema.
   - La tasa de aprendizaje fija Alpha = 0.005 ya es adecuada para GD simple.
   """)

print("3. Hiperparámetros:")
print(f"   - Para GD: valores mayores de Alpha convergen más rápido pero pueden oscilar.")
print(f"   - Para GDM: valores altos de Gamma (aprox. 0.9) suelen funcionar mejor en")
print(f"     funciones con múltiples mínimos locales como Himmelblau.")
print("="*70)
print()
# ---------------------------------------------------------------------------------------------------------------

# Pregunta 2
print("Pregunta 2 - Función de Rosenbrock")
print()

print("Encontrar el mínimo de la función de Rosenbrock:")
print("f(x, y) = (1 - x)^2 + 100*(y - x^2)^2")
print("Punto inicial: (-1, -1)")
print()

# Funcion de Rosenbrock
def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

# Gradiente de la funcion de Rosenbrock
def grad_rosenbrock(x):
    dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    dy = 200*(x[1] - x[0]**2)
    return np.array([dx, dy])

# Decaimiento por tiempo: alpha_k = alpha_0/(1 + k*tau)
def dg_tiempo(x_actual, alpha_0, tau, max_iter, tol):
    x_historial = [x_actual.copy()]
    f_historial = [rosenbrock(x_actual)]
    alpha_historial = [alpha_0]
    for k in range(max_iter):
        alpha_k = alpha_0/(1 + k*tau)
        alpha_historial.append(alpha_k)
        grad = grad_rosenbrock(x_actual)
        x_nuevo = x_actual - alpha_k*grad
        x_historial.append(x_nuevo.copy())
        f_historial.append(rosenbrock(x_nuevo))
        criterio_grad = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_cambio = np.linalg.norm(x_nuevo - x_actual)
        if criterio_grad < tol or criterio_cambio < tol:
            break
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(f_historial), np.array(alpha_historial)

# Decaimiento por pasos: alpha_k = alpha_0*d^(floor(k/s))
def dg_pasos(x_actual, alpha_0, d, s, max_iter, tol):
    x_historial = [x_actual.copy()]
    f_historial = [rosenbrock(x_actual)]
    alpha_historial = [alpha_0]
    for k in range(max_iter):
        alpha_k = alpha_0*(d**(k // s))  # Floor division
        alpha_historial.append(alpha_k)
        grad = grad_rosenbrock(x_actual)
        x_nuevo = x_actual - alpha_k*grad
        x_historial.append(x_nuevo.copy())
        f_historial.append(rosenbrock(x_nuevo))
        criterio_grad = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_cambio = np.linalg.norm(x_nuevo - x_actual)
        if criterio_grad < tol or criterio_cambio < tol:
            break
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(f_historial), np.array(alpha_historial)

# Decaimiento exponencial: alpha_k = alpha_0*e^(-k*tau)
def dg_exponencial(x_actual, alpha_0, tau, max_iter, tol):
    x_historial = [x_actual.copy()]
    f_historial = [rosenbrock(x_actual)]
    alpha_historial = [alpha_0]
    for k in range(max_iter):
        alpha_k = alpha_0*np.exp(-k*tau)
        alpha_historial.append(alpha_k)
        grad = grad_rosenbrock(x_actual)
        x_nuevo = x_actual - alpha_k*grad
        x_historial.append(x_nuevo.copy())
        f_historial.append(rosenbrock(x_nuevo))
        criterio_grad = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_cambio = np.linalg.norm(x_nuevo - x_actual)
        if criterio_grad < tol or criterio_cambio < tol:
            break
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(f_historial), np.array(alpha_historial)

# Descenso de gradiente con tasa de aprendizaje fija
def dg_fijo(x_actual, alpha, max_iter, tol):
    x_historial = [x_actual.copy()]
    f_historial = [rosenbrock(x_actual)]
    alpha_historial = [alpha]
    for k in range(max_iter):
        grad = grad_rosenbrock(x_actual)
        x_nuevo = x_actual - alpha*grad
        x_historial.append(x_nuevo.copy())
        f_historial.append(rosenbrock(x_nuevo))
        alpha_historial.append(alpha)
        criterio_grad = np.linalg.norm(grad_rosenbrock(x_nuevo))
        criterio_cambio = np.linalg.norm(x_nuevo - x_actual)
        if criterio_grad < tol or criterio_cambio < tol:
            break
        x_actual = x_nuevo
    
    return x_nuevo, np.array(x_historial), np.array(f_historial), np.array(alpha_historial)

# Experimentacion

# Parametros iniciales
x_inicial_ros = np.array([-1, -1])
alpha_0 = 0.002
max_iter = 10000
tol = 1e-6

# Parametros de decaimiento
tau = alpha_0/max_iter
d = 0.5
s = 2000

# Ejecutar los algoritmos
r_fijo = dg_fijo(x_inicial_ros.copy(), alpha_0, max_iter, tol)
r_tiempo = dg_tiempo(x_inicial_ros.copy(), alpha_0, tau, max_iter, tol)
r_pasos = dg_pasos(x_inicial_ros.copy(), alpha_0, d, s, max_iter, tol)
r_exponencial = dg_exponencial(x_inicial_ros.copy(), alpha_0, tau, max_iter, tol)

# Tabla comparativa
resultados = {'Estrategia': ['Alpha fijo',
                             'Decaimiento por tiempo',
                             'Decaimiento por pasos',
                             'Decaimiento exponencial'],
              'x_final': [f"({r_fijo[0][0]:.6f}, {r_fijo[0][1]:.6f})",
                          f"({r_tiempo[0][0]:.6f}, {r_tiempo[0][1]:.6f})",
                          f"({r_pasos[0][0]:.6f}, {r_pasos[0][1]:.6f})",
                          f"({r_exponencial[0][0]:.6f}, {r_exponencial[0][1]:.6f})"],
              'f(x,y)_final': [f"{r_fijo[2][-1]:.6e}",
                               f"{r_tiempo[2][-1]:.6e}",
                               f"{r_pasos[2][-1]:.6e}",
                               f"{r_exponencial[2][-1]:.6e}"],
              'Iteraciones': [len(r_fijo[1]) - 1,
                              len(r_tiempo[1]) - 1,
                              len(r_pasos[1]) - 1,
                              len(r_exponencial[1]) - 1]
}

df = pd.DataFrame(resultados)

print("Tabla comparativa")
print(df.to_string(index=False))
print("-"*70)

# Visualizaciones

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grafico de evolucion de f(x, y)
ax1 = axes[0]
ax1.semilogy(range(len(r_fijo[2])), r_fijo[2], 
             label='Alpha fijo', linewidth=2, alpha=0.8)
ax1.semilogy(range(len(r_tiempo[2])), r_tiempo[2], 
             label='Decaimiento por tiempo', linewidth=2, alpha=0.8)
ax1.semilogy(range(len(r_pasos[2])), r_pasos[2], 
             label='Decaimiento por pasos', linewidth=2, alpha=0.8)
ax1.semilogy(range(len(r_exponencial[2])), r_exponencial[2], 
             label='Decaimiento exponencial', linewidth=2, alpha=0.8)

ax1.set_xlabel('Iteración', fontsize=11)
ax1.set_ylabel('f(x, y) [escala logarítmica]', fontsize=11)
ax1.set_title('Evolución del valor de la función de Rosenbrock', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=9)
ax1.grid(True, alpha=0.3)

# Grafico de evolucion de alpha_k
ax2 = axes[1]
iteraciones_max = max(len(r_tiempo[3]), len(r_pasos[3]), len(r_exponencial[3]))
ax2.plot(range(len(r_tiempo[3])), r_tiempo[3], 
         label='Decaimiento por tiempo', linewidth=2, alpha=0.8)
ax2.plot(range(len(r_pasos[3])), r_pasos[3], 
         label='Decaimiento por pasos', linewidth=2, alpha=0.8)
ax2.plot(range(len(r_exponencial[3])), r_exponencial[3], 
         label='Decaimiento exponencial', linewidth=2, alpha=0.8)

ax2.set_xlabel('Iteración', fontsize=11)
ax2.set_ylabel('Tasa de aprendizaje (alpha_k)', fontsize=11)
ax2.set_title('Evolución de la tasa de aprendizaje', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Conclusiones
print("Conclusiones")
print()
print("1. Convergencia:")
print(f"   - Alpha fijo: {len(r_fijo[1])-1} iteraciones")
print(f"   - Decaimiento por tiempo: {len(r_tiempo[1])-1} iteraciones")
print(f"   - Decaimiento por pasos: {len(r_pasos[1])-1} iteraciones")
print(f"   - Decaimiento exponencial: {len(r_exponencial[1])-1} iteraciones")
print()
print("2. Mejor estrategia:")
mejor_idx = np.argmin([len(r_fijo[1]), len(r_tiempo[1]), len(r_pasos[1]), len(r_exponencial[1])])
estrategias_nombres = ['Alpha fijo', 'Decaimiento por tiempo', 'Decaimiento por pasos', 'Decaimiento exponencial']
print("   - Luego de la comparativa, se pudo determinar que la mejor estrategia de convergencia")
print(f"     es por {estrategias_nombres[mejor_idx]}.")
print()
print("3. Análisis:")
print("""   - Para la función de Rosenbrock con alpha_0 = 0.002, una tasa constante o
     decaimiento suave funciona bien.

   - El decaimiento por pasos es demasiado agresivo: reduce Alpha en un 50%
     cada 2000 iteraciones, volviéndose demasiado pequeño prematuramente.

   - El decaimiento por tiempo y exponencial con Tau = 2e-7 son tan suaves que es prácticamente
     equivalente a Alpha fijo durante las casi 8000 iteraciones.""")
print("="*70)
print()
# ---------------------------------------------------------------------------------------------------------------

#Pregunta 3
print("Pregunta 3 - Regresión con MAE")
print()

frecuencia_reloj = np.array([4.29, 4.34, 2.72, 4.08, 3.60,
                             3.30, 3.84, 2.34, 3.64, 3.76,
                             3.14, 3.80, 4.34, 2.64, 3.16,
                             4.35, 4.45, 2.29, 3.19, 3.40,
                             4.26, 2.35, 4.47, 4.37, 2.21])
consumo_energia = np.array([122, 130, 88, 121, 108,
                            102, 107, 75, 105, 119,
                            102, 112, 125, 79, 98,
                            127, 121, 73, 105, 101,
                            117, 78, 123, 119, 70])

# Outliers
frecuencia_reloj = np.append(frecuencia_reloj, [2.5, 4.5])
consumo_energia = np.append(consumo_energia, [130, 70])

print("Datos con outliers:")
print(f"Número total de puntos: {len(frecuencia_reloj)}")
print(f"Outliers añadidos: (2.5, 130) y (4.5, 70)")
print("-"*70)
print()

# Modelo MSE

# Regresion lineal usando minimos cuadrados
def regre_mse(X, y):
    # Añadir columna de unos para Beta_0 (intercepto)
    X_b = np.c_[np.ones((len(X), 1)), X]
    # Solución de forma cerrada: Beta = (X^T X)^-1 X^T y
    beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return beta[0], beta[1]  # Beta_0, Beta_1

# Modelo MAE

# Funcion objetivo MAE
def mae_objetivo(X, y, beta_0, beta_1):
    prediccion = beta_0 + beta_1 * X
    return np.mean(np.abs(y - prediccion))

# Subgradiente de MAE
def sgrad_mae(X, y, beta_0, beta_1):
    prediccion = beta_0 + beta_1 * X
    residuo = y - prediccion
    
    # Funcion sign: -1 si z < 0, +1 si z > 0, 0 si z = 0
    signs = np.sign(residuo)
    
    # Subgradiente para beta_0 (x_i0 = 1 para todos)
    grad_beta_0 = -np.mean(signs)
    
    # Subgradiente para beta_1
    grad_beta_1 = -np.mean(X * signs)
    
    return grad_beta_0, grad_beta_1

# Regresion lineal con descenso de subgradiente
def regre_mae(X, y, alpha = 0.01, max_iter = 10000, tol = 1e-6):
    # Inicialización
    beta_0 = 0
    beta_1 = 0
    
    for iteration in range(max_iter):
        # Calcular subgradiente
        grad_0, grad_1 = sgrad_mae(X, y, beta_0, beta_1)
        
        # Guardar valores anteriores
        beta_0_old = beta_0
        beta_1_old = beta_1
        
        # Actualizar parámetros
        beta_0 = beta_0 - alpha*grad_0
        beta_1 = beta_1 - alpha*grad_1
        
        # Criterio de convergencia
        convergencia = np.sqrt((beta_0 - beta_0_old)**2 + (beta_1 - beta_1_old)**2)
        if convergencia < tol:
            print(f"MAE convergió en {iteration + 1} iteraciones")
            break
    
    return beta_0, beta_1

# Ejecucion de modelos
print("Resultados de la regresión")
print()

# Modelo MSE
beta_0_mse, beta_1_mse = regre_mse(frecuencia_reloj, consumo_energia)
y_pred_mse = beta_0_mse + beta_1_mse*frecuencia_reloj
mse_error = np.mean((consumo_energia - y_pred_mse)**2)

print("Modelo MSE (Mínimos Cuadrados):")
print(f"   Beta_0 (intercepto) = {beta_0_mse:.4f}")
print(f"   Beta_1 (pendiente)  = {beta_1_mse:.4f}")
print(f"   Ecuación: y = {beta_0_mse:.4f} + {beta_1_mse:.4f}x")
print(f"   MSE = {mse_error:.4f}")
print()

# Modelo MAE
beta_0_mae, beta_1_mae = regre_mae(frecuencia_reloj, consumo_energia, 
                                            alpha=0.1, max_iter = 10000)
y_pred_mae = beta_0_mae + beta_1_mae*frecuencia_reloj
mae_error = np.mean(np.abs(consumo_energia - y_pred_mae))

print("Modelo MAE (Robusto):")
print(f"   Beta_0 (intercepto) = {beta_0_mae:.4f}")
print(f"   Beta_1 (pendiente)  = {beta_1_mae:.4f}")
print(f"   Ecuación: y = {beta_0_mae:.4f} + {beta_1_mae:.4f}x")
print(f"   MAE = {mae_error:.4f}")
print("-"*70)

# Visualizacion
plt.figure(figsize=(12, 8))

# Separar datos originales y outliers para la visualización
X_original = frecuencia_reloj[:-2]
y_original = consumo_energia[:-2]
X_outliers = frecuencia_reloj[-2:]
y_outliers = consumo_energia[-2:]

# Scatter plot
plt.scatter(X_original, y_original, s = 100, alpha = 0.6, edgecolors = 'black', 
            linewidth = 1.5, label = 'Datos originales', zorder = 3)
plt.scatter(X_outliers, y_outliers, s = 200, c = 'red', marker = 'X', 
            edgecolors = 'darkred', linewidth = 2, label = 'Outliers', zorder = 4)

# Lineas de regresion
x_line = np.linspace(frecuencia_reloj.min() - 0.2, frecuencia_reloj.max() + 0.2, 100)
y_line_mse = beta_0_mse + beta_1_mse*x_line
y_line_mae = beta_0_mae + beta_1_mae*x_line

plt.plot(x_line, y_line_mse, 'b--', linewidth = 2.5, 
         label = f'MSE: y = {beta_0_mse:.2f} + {beta_1_mse:.2f}x', alpha = 0.8)
plt.plot(x_line, y_line_mae, 'g-', linewidth = 2.5, 
         label = f'MAE: y = {beta_0_mae:.2f} + {beta_1_mae:.2f}x', alpha = 0.8)

plt.xlabel('Frecuencia de Reloj (GHz)', fontsize = 12, fontweight = 'bold')
plt.ylabel('Consumo de Energía (W)', fontsize = 12, fontweight = 'bold')
plt.title('Comparación: Regresión MSE vs MAE con Outliers', fontsize = 14, fontweight = 'bold', pad = 20)
plt.legend(fontsize = 10, loc = 'best', framealpha = 0.9)
plt.grid(True, alpha = 0.3, linestyle = '--')
plt.tight_layout()
plt.show()

# Analisis
print("Análisis y conclusiones")

print("""
El modelo basado en MAE es considerado más ROBUSTO ante valores atípicos por
las siguientes razones:

1. Penalización de errores:
   • MSE penaliza los errores con el cuadrado: error^2
     - Un error de 50 unidades → penalización de 2500
   • MAE penaliza los errores linealmente: |error|
     - Un error de 50 unidades → penalización de 50
   
   Los outliers generan errores GRANDES que dominan el MSE, forzando al modelo
   a ajustarse hacia ellos.

2. Impacto en la regresión:
   • MSE: La recta se "tuerce" significativamente para reducir los errores^2
     de los outliers, alejándose de la tendencia general de los datos.
   • MAE: La recta mantiene la tendencia general porque los outliers no
     tienen un impacto desproporcionado.

3. Ajuste a la tendencia general:
   La recta del modelo MAE se ajusta MEJOR a la tendencia de los datos
   originales (sin considerar los outliers), mientras que la recta MSE
   se ve "atraída" hacia los valores atípicos.

4. Robustez estadística:
   MAE es una métrica de tendencia central más robusta. Mientras que el
   promedio (usado en MSE) es sensible a valores extremos, la mediana
   (relacionada con MAE) es resistente a outliers.

Conclusión:
Para datos con posibles errores de medición o valores atípicos, el modelo
basado en MAE proporciona una regresión más confiable y representativa de
la relación real entre las variables.""")
print("="*70)
