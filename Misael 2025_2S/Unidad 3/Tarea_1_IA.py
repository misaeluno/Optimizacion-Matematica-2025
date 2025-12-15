import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===================================================================
# DATOS SIMULADOS
# ===================================================================
np.random.seed(2025)
m = 100

# Variable independiente (usuarios)
usuarios = np.random.uniform(10, 100, m)

# Variable dependiente (peticiones)
lambda_real = np.exp(0.5 + 0.03 * usuarios)
peticiones = np.random.poisson(lambda_real)

datos = pd.DataFrame({"Usuarios": usuarios, "Peticiones": peticiones})
print("=" * 60)
print("DATOS SIMULADOS (primeras 10 observaciones)")
print("=" * 60)
print(datos.head(10))
print()

# ===================================================================
# FUNCIONES DEL MODELO
# ===================================================================

def calcular_lambda(beta0, beta1, x):
    """Calcula λ = exp(β₀ + β₁·x)"""
    return np.exp(beta0 + beta1 * x)

def funcion_costo(beta0, beta1, x, y):
    """
    Calcula J(β) = Σ(λ - y·log(λ))
    Esta es la log-verosimilitud negativa
    """
    lambda_vals = calcular_lambda(beta0, beta1, x)
    # Añadimos epsilon para evitar log(0)
    epsilon = 1e-10
    costo = np.sum(lambda_vals - y * np.log(lambda_vals + epsilon))
    return costo

def calcular_gradiente(beta0, beta1, x, y):
    """
    Calcula el gradiente ∇J(β)
    
    ∂J/∂β₀ = Σ(λ - y)
    ∂J/∂β₁ = Σ x·(λ - y)
    """
    lambda_vals = calcular_lambda(beta0, beta1, x)
    error = lambda_vals - y
    
    grad_beta0 = np.sum(error)
    grad_beta1 = np.sum(x * error)
    
    return grad_beta0, grad_beta1

# ===================================================================
# DESCENSO DE GRADIENTE
# ===================================================================

def descenso_gradiente(x, y, alpha=0.00001, iteraciones=10000, tolerancia=1e-8):
    """
    Implementa descenso de gradiente para encontrar β óptimos
    
    Parámetros:
    - x: variable independiente (usuarios)
    - y: variable dependiente (peticiones)
    - alpha: tasa de aprendizaje
    - iteraciones: número máximo de iteraciones
    - tolerancia: criterio de convergencia
    """
    # Inicialización
    beta0 = 0.0
    beta1 = 0.0
    
    historial_costo = []
    historial_beta0 = []
    historial_beta1 = []
    
    print("=" * 60)
    print("DESCENSO DE GRADIENTE")
    print("=" * 60)
    print(f"Tasa de aprendizaje (α): {alpha}")
    print(f"Iteraciones máximas: {iteraciones}")
    print(f"Tolerancia: {tolerancia}")
    print()
    
    for i in range(iteraciones):
        # Guardar valores actuales
        costo_actual = funcion_costo(beta0, beta1, x, y)
        historial_costo.append(costo_actual)
        historial_beta0.append(beta0)
        historial_beta1.append(beta1)
        
        # Calcular gradiente
        grad_beta0, grad_beta1 = calcular_gradiente(beta0, beta1, x, y)
        
        # Actualizar parámetros
        beta0_nuevo = beta0 - alpha * grad_beta0
        beta1_nuevo = beta1 - alpha * grad_beta1
        
        # Imprimir progreso cada 1000 iteraciones
        if i % 1000 == 0:
            print(f"Iteración {i:5d} | Costo: {costo_actual:10.4f} | "
                  f"β₀: {beta0:8.5f} | β₁: {beta1:8.5f}")
        
        # Verificar convergencia
        cambio_beta0 = abs(beta0_nuevo - beta0)
        cambio_beta1 = abs(beta1_nuevo - beta1)
        
        if cambio_beta0 < tolerancia and cambio_beta1 < tolerancia:
            print(f"\n¡Convergencia alcanzada en iteración {i}!")
            break
        
        beta0 = beta0_nuevo
        beta1 = beta1_nuevo
    
    return beta0, beta1, historial_costo, historial_beta0, historial_beta1

# Ejecutar descenso de gradiente
x = usuarios
y = peticiones

beta0_opt, beta1_opt, hist_costo, hist_beta0, hist_beta1 = descenso_gradiente(
    x, y, alpha=0.00001, iteraciones=10000
)

print()
print("=" * 60)
print("RESULTADOS FINALES")
print("=" * 60)
print(f"Parámetros reales:    β₀ = 0.5000, β₁ = 0.0300")
print(f"Parámetros estimados: β₀ = {beta0_opt:.4f}, β₁ = {beta1_opt:.4f}")
print()

# ===================================================================
# PREDICCIONES
# ===================================================================

# Predicción para carga inédita
usuarios_nuevos = np.array([20, 50, 80, 120])

print("=" * 60)
print("PREDICCIONES PARA CARGAS INÉDITAS")
print("=" * 60)

for u in usuarios_nuevos:
    lambda_pred = calcular_lambda(beta0_opt, beta1_opt, u)
    print(f"Usuarios: {u:3d} → Tasa predicha (λ): {lambda_pred:8.2f} peticiones/min")

print()

# ===================================================================
# VISUALIZACIONES
# ===================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Convergencia del costo
axes[0, 0].plot(hist_costo, color='red', linewidth=2)
axes[0, 0].set_xlabel('Iteración')
axes[0, 0].set_ylabel('Costo J(β)')
axes[0, 0].set_title('Convergencia de la Función de Costo')
axes[0, 0].grid(True, alpha=0.3)

# 2. Convergencia de parámetros
axes[0, 1].plot(hist_beta0, label='β₀', linewidth=2)
axes[0, 1].plot(hist_beta1, label='β₁', linewidth=2)
axes[0, 1].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='β₀ real')
axes[0, 1].axhline(y=0.03, color='orange', linestyle='--', alpha=0.5, label='β₁ real')
axes[0, 1].set_xlabel('Iteración')
axes[0, 1].set_ylabel('Valor del parámetro')
axes[0, 1].set_title('Convergencia de Parámetros')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Ajuste del modelo
axes[1, 0].scatter(usuarios, peticiones, alpha=0.5, label='Datos observados')
x_plot = np.linspace(10, 100, 200)
y_plot = calcular_lambda(beta0_opt, beta1_opt, x_plot)
axes[1, 0].plot(x_plot, y_plot, 'r-', linewidth=2, label='Modelo ajustado')
axes[1, 0].set_xlabel('Usuarios activos')
axes[1, 0].set_ylabel('Peticiones (hits/min)')
axes[1, 0].set_title('Ajuste del Modelo GLM Poisson')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuos
lambda_ajustado = calcular_lambda(beta0_opt, beta1_opt, usuarios)
residuos = peticiones - lambda_ajustado
axes[1, 1].scatter(lambda_ajustado, residuos, alpha=0.5)
axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('λ predicho')
axes[1, 1].set_ylabel('Residuos (y - λ)')
axes[1, 1].set_title('Análisis de Residuos')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===================================================================
# MÉTRICAS DE EVALUACIÓN
# ===================================================================
print("=" * 60)
print("MÉTRICAS DE EVALUACIÓN")
print("=" * 60)

# Error cuadrático medio
mse = np.mean((peticiones - lambda_ajustado) ** 2)
rmse = np.sqrt(mse)
print(f"Error Cuadrático Medio (MSE): {mse:.2f}")
print(f"Raíz del MSE (RMSE): {rmse:.2f}")

# R² (pseudo R² para Poisson)
ss_res = np.sum((peticiones - lambda_ajustado) ** 2)
ss_tot = np.sum((peticiones - np.mean(peticiones)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"Pseudo R²: {r2:.4f}")
print("=" * 60)