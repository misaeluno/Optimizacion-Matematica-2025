# Cargar librerías necesarias
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
# Configuraciones iniciales
# -------------------------
plt.rcParams["figure.dpi"]

# Datos
# -----
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

# Diagrama de dispersión
# ----------------------
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(frecuencia_reloj, consumo_energia, "ro")
ax.set_title("Consumo de energía vs. frecuencia")
ax.set_xlabel("Frecuencia de reloj (GHz)")
ax.set_ylabel("Consumo de energía (Watt)")
plt.show()

# Variables para el modelo
# ------------------------
x = frecuencia_reloj
y = consumo_energia
# Función objetivo a minimizar
# ----------------------------
def f_objetivo(beta, x, y):
    error = y - (beta[0] + beta[1] * x)
    return np.sum(error ** 2)
# Gradiente de la función objetivo
# --------------------------------
def g_objetivo(beta, x, y):
    error = y - (beta[0] + beta[1] * x)
    d_beta_0 = -2 * np.sum(error)
    d_beta_1 = -2 * np.sum(error * x)
    return np.array([d_beta_0, d_beta_1])

# Hessiana de la función objetivo
# -------------------------------
def h_objetivo(beta, x, y):
    n = len(x)
    d00 = 2 * n
    d01 = 2 * np.sum(x)
    d11 = 2 * np.sum(x ** 2)
    return np.array([[d00, d01], [d01, d11]])

# Algoritmo de descenso de gradiente
# ----------------------------------
def dg_rls(beta_actual, x, y, alpha, max_iter = 10000, tol = 1e-6):
    beta_historial = [beta_actual]
    for _ in range(max_iter):
        beta_nuevo = beta_actual - alpha * g_objetivo(beta_actual, x, y)
        beta_historial.append(beta_nuevo)
        criterio_1 = np.linalg.norm(beta_nuevo - beta_actual)
        criterio_2 = np.linalg.norm(g_objetivo(beta_nuevo, x, y))
        if criterio_1 < tol or criterio_2 < tol:
            break
        beta_actual = beta_nuevo
    beta_historial = np.array(beta_historial)
    return beta_nuevo, beta_historial

# Parámetros y ejecución
alpha_dg = 0.001
beta_inicial = np.array([0.0, 1.0])
beta_dg, hist_dg = dg_rls(beta_inicial, x, y, alpha_dg)
print("--- Descenso de gradiente ---")
print("Beta estimado =", beta_dg)
print("Número de iteraciones =", len(hist_dg))

# Algoritmo de Newton
# -------------------
def newton_rls(beta_actual, x, y, max_iter = 10000, tol = 1e-6):
    beta_historial = [beta_actual]
    for _ in range(max_iter):
        beta_nuevo = beta_actual - np.linalg.inv(h_objetivo(beta_actual, x, y)) @ g_objetivo(beta_actual, x, y)
        beta_historial.append(beta_nuevo)
        criterio_1 = np.linalg.norm(beta_nuevo - beta_actual)
        criterio_2 = np.linalg.norm(g_objetivo(beta_nuevo, x, y))
        if criterio_1 < tol or criterio_2 < tol:
            break
        beta_actual = beta_nuevo
    beta_historial = np.array(beta_historial)
    return beta_nuevo, beta_historial
# Parámetros y ejecución
# ----------------------
beta_inicial = np.array([0.0, 1.0])
beta_newton, hist_newton = newton_rls(beta_inicial, x, y)
print("--- Método de Newton ---")
print("Beta estimado =", beta_newton)
print("Número de iteraciones =", len(hist_newton))

# Algoritmo Quasi-Newton
# ----------------------
def h_bfgs(h_actual, s_actual, y_actual):
    yts_actual = np.dot(y_actual, s_actual)
    if yts_actual <= 1e-10:
        return h_actual
    n = s_actual.shape[0]
    I = np.identity(n)
    aux1 = I - np.outer(s_actual, y_actual) / yts_actual
    aux2 = I - np.outer(y_actual, s_actual) / yts_actual
    aux3 = np.outer(s_actual, s_actual) / yts_actual
    h_nuevo = aux1 @ h_actual @ aux2 + aux3
    return h_nuevo

def backtracking(f, grad_f, x_actual, p_actual, alpha_inicial = 1.0, rho = 0.5,c = 1e-4):
    alpha = alpha_inicial
    gradiente_actual = grad_f(x_actual, x, y)
    while f(x_actual + alpha * p_actual, x, y) > f(x_actual, x, y) + c * alpha* np.dot(gradiente_actual, p_actual):
        alpha = alpha * rho
    return alpha

def qn_rls(f, grad_f, beta_actual, x, y, max_iter = 10000, tol = 1e-6):
    beta_historial = [beta_actual]
    h_actual = np.identity(len(beta_actual))
    for i in range(max_iter):
        grad_actual = grad_f(beta_actual, x, y)
        p_actual = -h_actual @ grad_actual
        alpha_actual = backtracking(f, grad_f, beta_actual, p_actual)
        beta_nuevo = beta_actual + alpha_actual * p_actual
        beta_historial.append(beta_nuevo)
        criterio_1 = np.linalg.norm(beta_nuevo - beta_actual)
        criterio_2 = np.linalg.norm(g_objetivo(beta_nuevo, x, y))
        if criterio_1 < tol or criterio_2 < tol:
            break
        grad_nuevo = grad_f(beta_nuevo, x, y)
        s_actual = beta_nuevo - beta_actual
        y_actual = grad_nuevo - grad_actual
        h_actual = h_bfgs(h_actual, s_actual, y_actual)
        beta_actual = beta_nuevo
    beta_historial = np.array(beta_historial)
    return beta_nuevo, beta_historial

 # Ejecución
beta_inicial = np.array([0.0, 1.0])
beta_qn, hist_qn = qn_rls(f_objetivo, g_objetivo, beta_inicial, x, y)
print("--- Método de Quasi-Newton ---")
print("Beta estimado =", beta_qn)
print("Número de iteraciones =", len(hist_qn))   

# Gráfica de la recta de regresión
# --------------------------------
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(frecuencia_reloj, consumo_energia, "ro")
x_regresion = np.linspace(min(frecuencia_reloj), max(frecuencia_reloj), 100)
y_regresion = beta_newton[0] + beta_newton[1] * x_regresion
ax.plot(x_regresion, y_regresion, label = "Modelo de regresión")
ax.set_title("Consumo de energía vs. frecuencia")
ax.set_xlabel("Frecuencia de reloj (GHz)")
ax.set_ylabel("Consumo de energía (Watt)")
ax.legend()
plt.show()