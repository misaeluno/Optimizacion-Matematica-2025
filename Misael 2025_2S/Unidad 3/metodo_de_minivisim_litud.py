# Cargar librerías necesarias
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import weibull_min
# Configuraciones iniciales
# -------------------------

# Generación de datos simulados (logs del servidor)
# ------------------------------------------------
np.random.seed(2025)
# Parámetros reales
k_verdadero = 1.5
lambda_verdadero = 500
n_muestras = 500
# Generamos tiempos de falla (datos observables)
tiempos_falla = weibull_min.rvs(k_verdadero, scale = lambda_verdadero, size =n_muestras)
print(f"Datos generados: {n_muestras} observaciones.")
print(f"Ejemplos de tiempos de falla en horas: {tiempos_falla[:5]}")

# Histograma de los tiempos de falla
# ----------------------------------
fig, ax = plt.subplots(figsize = (6, 3))
ax.hist(tiempos_falla, bins = 30, density = True,
        color = "gray", edgecolor = "black",
        label = "Logs del servidor")
ax.set_title("Histograma de tiempo de falla")
ax.set_xlabel("Tiempos de falla en horas")
ax.set_ylabel("Densidad")
ax.legend()
plt.show()

# Definición del modelo de optimización
# -------------------------------------
def menos_log_verosimilitud(parametros, tiempos):
    k = np.exp(parametros[0])
    lam = np.exp(parametros[1])
    n = len(tiempos)
    sum_log_t = np.sum(np.log(tiempos))
    sum_t_lambda = np.sum((tiempos / lam) ** k)
    mlv = -n * np.log(k) + n * k * np.log(lam) - (k - 1) * sum_log_t + sum_t_lambda
    return mlv


# Proceso de optimización
# -----------------------
parametros_inicial = np.array([0.0, 0.0]) # Corresponde a k = 1 y lambda = 1
resultado = minimize(fun = menos_log_verosimilitud,
x0 = parametros_inicial,
args = (tiempos_falla),
method = "Nelder-Mead")
# Recuperar parámetros
k_estimado = np.exp(resultado.x[0])
lambda_estimado = np.exp(resultado.x[1])
print("--- Resultados de la Estimación ---")
print(f"Éxito: {resultado.success}")
print(f"Iteraciones: {resultado.nit}")
print(f"Parámetros Reales: k = {k_verdadero:.4f}, lambda = {lambda_verdadero:.4f}")
print(f"Parámetros Estimados: k = {k_estimado:.4f}, lambda = {lambda_estimado:.4f}")

# Ajuste de densidad de probabilidad de fallos
# -------------------------------------------
fig, ax = plt.subplots(figsize = (6, 3))
ax.hist(tiempos_falla, bins = 30, density = True,
color = "gray", edgecolor = "black",
label = "Logs del servidor")
x_ajuste = np.linspace(0, max(tiempos_falla), 1000)
y_ajuste = weibull_min.pdf(x_ajuste, k_estimado, scale = lambda_estimado)
ax.plot(x_ajuste, y_ajuste, 'r-', lw = 4,
label = f"Ajuste Weibull (k = {k_estimado:.2f})")
ax.set_title("Ajuste de densidad de probabilidad de fallos")
ax.set_xlabel("Tiempos de falla en horas")
ax.set_ylabel("Densidad")
ax.legend()
plt.show()

# Función de Supervivencia (Reliability Function R(t))
# R(t) = 1 - CDF(t)
fig, ax = plt.subplots(figsize = (6, 3))
reliability = np.exp(-(x_ajuste / lambda_estimado) ** k_estimado)
ax.plot(x_ajuste, reliability, "b-", lw = 4)
ax.set_title("Curva de supervivencia $R(t)$")
ax.set_xlabel("Tiempo (horas)")
ax.set_ylabel("Probabilidad de supervivencia")
ax.grid(True, alpha = 0.25)
plt.show()