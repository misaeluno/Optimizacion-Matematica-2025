# Cargar librerías necesarias
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Configuraciones iniciales
# -------------------------
plt.rcParams["figure.dpi"]

# Generación de datos simulados
# -----------------------------
np.random.seed(2025)
# Simulamos 2 características:
# x1: Frecuencia de palabra clave (por ejemplo, "oferta", "gratis")
# x2: Cantidad de enlaces en el correo
# m: número de correos
m = 500
# Generamos correos No-spam (clase 0)
# Centrados en (2, 2) con dispersión
X_no_spam = np.random.randn(250, 2) + [2, 2]
y_no_spam = np.zeros(250)
# Generamos correos Spam (clase 1)
# Centrados en (4, 4) con dispersión
X_spam = np.random.randn(250, 2) + [4, 4]
y_spam = np.ones(250)
# Concatemanos los datos
X_brutos = np.vstack((X_no_spam, X_spam))
y = np.hstack((y_no_spam, y_spam))
# Añadimos el sesgo (intercepto): Columna de 1's al inicio
# Esto permite que la recta de separación no tenga que
# pasar por el origen (0, 0)
X = np.column_stack((np.ones(m), X_brutos))
print(f"Dimensiones de X: {X.shape}")

# Visualización de los datos
# --------------------------
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(X_brutos[:, 0], X_brutos[:, 1], color = "black")
ax.set_xlabel("Característica 1 (frecuencia palabras clave)")
ax.set_ylabel("Característica 2 (cantidad de enlaces)")
ax.set_title("Clasificación de spam mediante optimización BFGS")
plt.grid(True, alpha = 0.25)
plt.show()

# Visualización de los datos
# --------------------------
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(X_brutos[y == 0][:, 0], X_brutos[y == 0][:, 1], color = "blue", label = "No-spam (0)")
ax.scatter(X_brutos[y == 1][:, 0], X_brutos[y == 1][:, 1], color = "red", label= "Spam (1)")
ax.set_xlabel("Característica 1 (frecuencia palabras clave)")
ax.set_ylabel("Característica 2 (cantidad de enlaces)")
ax.set_title("Clasificación de spam mediante optimización BFGS")
ax.legend()
plt.grid(True, alpha = 0.25)
plt.show()

# Definición de funciones numéricas
# ---------------------------------
def sigmoide(z):
    # Clip para evitar desbordamiento numérico (overflow) en exp()
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def funcion_objetivo(beta, X, y):
    m = len(y)
    h = sigmoide(np.dot(X, beta))
    epsilon = 1e-5
    log_ver_neg = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return log_ver_neg

# Ejecución de la optimización
# ----------------------------
# Inicialización de los coeficienes (betas)
beta_inicial = np.zeros(X.shape[1])
# Ejecución
resultado = minimize(fun = funcion_objetivo,
                     x0 = beta_inicial,
                    args = (X, y),
                    method = "BFGS")
beta_optimo = resultado.x
print(resultado)

# Resultados
print("=== RESULTADOS ===")
print(f"Costo final: {resultado.fun:.4f}")
print(f"Coeficientes óptimos (beta): {beta_optimo}")

# Visualización de la frontera de decisión
fig, ax = plt.subplots(figsize = (10, 6))
ax.scatter(X_brutos[y == 0][:, 0], X_brutos[y == 0][:, 1], color = "blue", label = "No-spam (0)")
ax.scatter(X_brutos[y == 1][:, 0], X_brutos[y == 1][:, 1], color = "red", label = "Spam (1)")
# Calcular la recta de decisión
# La frontera es donde h(x) = 0.5 => beta0 + beta1 * x1 + beta2 * x2 = 0
# Despejamos x2: x2 = -(beta0 + beta1 * x1) / beta2
x1_valores = np.array([np.min(X_brutos[:, 0]), np.max(X_brutos[:, 0])])
x2_valores = -(beta_optimo[0] + beta_optimo[1] * x1_valores) / beta_optimo[2]
ax.plot(x1_valores, x2_valores, "k--", linewidth=2, label = "Frontera de decisión (BFGS)")
ax.set_xlabel("Característica 1 (frecuencia palabras clave)")
ax.set_ylabel("Característica 2 (cantidad de enlaces)")
ax.set_title("Clasificación de spam mediante optimización BFGS")
ax.legend()
ax.grid(True, alpha = 0.25)
plt.show()