# Cargar librerías necesarias
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
# Configuraciones inciales
# ------------------------
plt.rcParams["figure.dpi"] 

# Datos
# -----
x = np.array([0.0, 0.5, 1.0, 1.5, 2.0,
            2.5, 3.0, 3.5, 4.0, 4.5,
            5.0, 5.5, 6.0, 6.5, 7.0,
            7.5, 8.0, 8.5, 9.0, 9.5,
            10.0, 10.5, 11.0, 11.5, 12.0,
            12.5, 13.0, 13.5, 14.0, 14.5,
            15.0, 15.5, 16.0, 16.5, 17.0,
            17.5, 18.0, 18.5, 19.0, 19.5,
            20.0, 20.5, 21.0, 21.5, 22.0,
            22.5, 23.0, 23.5])
y = np.array([58.319, 60.268, 66.515, 62.859, 63.780,
            69.223, 66.096, 61.508, 63.795, 65.041,
            70.547, 68.434, 69.026, 68.621, 67.089,
            74.595, 71.222, 64.345, 72.895, 69.956,
            68.797, 72.022, 70.331, 72.019, 73.197,
            70.956, 79.556, 78.621, 75.962, 84.457,
            83.404, 82.787, 88.029, 89.781, 91.552,
            93.238, 95.070, 95.616, 97.443, 99.950,
            101.916, 106.470, 106.583, 120.370,
            121.176, 118.084, 124.364, 128.518])

# Diagrama de dispersión
# ----------------------
fig, ax = plt.subplots(figsize = (6, 3))
ax.plot(x, y, "ro")
ax.set_title("Temperatura del reactor vs. tiempo")
ax.set_xlabel("Tiempo (horas)")
ax.set_ylabel("Temperatura (ºC)")
plt.show()

# ----------------------------
# Función objetivo a minimizar
# ----------------------------
def f_objetivo(beta, x, y):
    e = y - (beta[0] + beta[1] * x + beta[2] * x ** 2 + beta[3] * x ** 3)
    return np.sum(e ** 2)
# --------------------------------
# Gradiente de la función objetivo
# --------------------------------
def g_objetivo(beta, x, y):
    e = y - (beta[0] + beta[1] * x + beta[2] * x ** 2 + beta[3] * x ** 3)
    d0 = -2 * np.sum(e)
    d1 = -2 * np.sum(e * x)
    d2 = -2 * np.sum(e * x ** 2)
    d3 = -2 * np.sum(e * x ** 3)
    return np.array([d0, d1, d2, d3])

# -------------------------------
# Hessiana de la función objetivo
# -------------------------------
def h_objetivo(beta, x, y):
    e = y - (beta[0] + beta[1] * x + beta[2] * x ** 2 + beta[3] * x ** 3)
    n = len(x)
    d00 = 2 * n
    d01 = d10 = 2 * np.sum(x)
    d02 = d20 = 2 * np.sum(x ** 2)
    d03 = d30 = 2 * np.sum(x ** 3)
    d11 = 2 * np.sum(x ** 2)
    d12 = d21 = 2 * np.sum(x ** 3)
    d13 = d31 = 2 * np.sum(x ** 4)
    d22 = 2 * np.sum(x ** 4)
    d23 = d32 = 2 * np.sum(x ** 5)
    d33 = 2 * np.sum(x ** 6)
    return np.array([[d00, d01, d02, d03],
                    [d10, d11, d12, d13],
                    [d20, d21, d22, d23],
                    [d30, d31, d32, d33]])
# ----------------------------------
# Algoritmo de descenso de gradiente
# ----------------------------------
def dg_rp(beta_actual, x, y, alpha, max_iter = 100000, tol = 1e-6):
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
# alpha_dg = 0.001
# beta_inicial = np.array([60, 2, 0, 0])
# beta_dg, hist_dg = dg_rp(beta_inicial, x, y, alpha_dg)
# print("--- Descenso de gradiente ---")
# print("Beta estimado =", beta_dg)
# print("Número de iteraciones =", len(hist_dg))

# -------------------
# Algoritmo de Newton
# -------------------
def newton_rp(beta_actual, x, y, max_iter = 10000, tol = 1e-6):
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
beta_inicial = np.array([60, 2, 0, 0])
beta_newton, hist_newton = newton_rp(beta_inicial, x, y)
print("--- Método de Newton ---")
print("Beta estimado =", beta_newton)
print("Número de iteraciones =", len(hist_newton))

# ----------------------
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

def backtracking(f, grad_f, x_actual, p_actual, alpha_inicial = 1.0, rho = 0.5, c = 1e-4):
    alpha = alpha_inicial
    gradiente_actual = grad_f(x_actual, x, y)
    while f(x_actual + alpha * p_actual, x, y) > f(x_actual, x, y) + c * alpha * np.dot(gradiente_actual, p_actual):
        alpha = alpha * rho
    return alpha

def qn_rp(f, grad_f, beta_actual, x, y, max_iter = 10000, tol = 1e-6):
    beta_historial = [beta_actual]
    h_actual = np.identity(len(beta_actual))
    for _ in range(max_iter):
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
beta_inicial = np.array([60, 2, 0, 0])
beta_qn, hist_qn = qn_rp(f_objetivo, g_objetivo, beta_inicial, x, y)
print("--- Método de Quasi-Newton ---")
print("Beta estimado =", beta_qn)
print("Número de iteraciones =", len(hist_qn))

# --------------------------------
# Gráfica de la recta de regresión
# --------------------------------
plt.figure(figsize=(6, 3))
plt.plot(x, y, "ro")
x_regresion = np.linspace(min(x), max(x), 100)
y_regresion = beta_newton[0] + beta_newton[1] * x_regresion + beta_newton[2] *x_regresion ** 2 + beta_newton[3] * x_regresion ** 3
plt.plot(x_regresion, y_regresion, label = "Modelo de regresión")
plt.xlabel("Tiempo (horas)")
plt.ylabel("Temperatura (ºC)")
plt.title("Temperatura del reactor vs. tiempo")
plt.tick_params(labelsize = 8)
plt.legend()
plt.show()

