import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import weibull_min

np.random.seed(2025)

#parametros reales (desconocidos para nosotros)
k_verdaderos = 1.5
lambda_verdadero = 500
n_muestra = 500

#generar los timepos de falla (datos observados)
tiempos_falla = weibull_min.rvs(k_verdaderos, scale = lambda_verdadero, size = n_muestra)

print(f"Datos generales: {n_muestra} observados")
print(f"ejemplos de tiempos de falla en horas: {tiempos_falla[:5]}")

#grafico de valores de tiempo de vida de servidores
fig, ax = plt.subplots(figsize = (6, 3))
ax.hist(tiempos_falla, bins = 30, density = True,
        color = "gray", edgecolor = "black",
        label = "logs de servidor")
ax.set_title("histograma de tiemppo de falla")
ax.set_xlabel("")
ax.set_ylabel("densidad")
plt.show()

#funcion objetivo
def menos_log_verosimilitud(parametros, tiempos):
    k = parametros[0]
    landa= parametros[1]
    n = len(tiempos)
    sum_log_t = np.sum(np.log(tiempos))
    sum_t_landa = np.sum((tiempos/1)**k)
    mlv = (-n*np.log(k)) + (n*k*np.log(landa)) - ((k - 1)*sum_log_t) + sum_t_landa
    return mlv

#estimacion de maxima verosimilitud
parametro_iniciales = np.array([1.0, 200])

print("iniciando optimizacion con nelder-mead...")
resultado = minimize(fun = menos_log_verosimilitud,
                     x0 = parametro_iniciales,
                     args = (tiempos_falla),
                     method = "Nelder-mead")

k_estimado = resultado.x[0]
lamda_estimado = resultado.x[1]
print(resultado)
print("---- resultado de la estimacion ---")
print(f"exito: {resultado.success}")
print(f"iteraciones {resultado.nit}")
print(f"parametros reales son:      k = {k_verdaderos:.4}, lamda = {lambda_verdadero:.4}")
print(f"parametros estimados son:   k = {k_estimado:.4}, lamda = {lamda_estimado:.4}")

#grafico anterior segun datos obtenidos
fig, ax = plt.subplots(figsize = (6, 3))
ax.hist(tiempos_falla, bins = 30, density = True,
        color = "gray", edgecolor = "black",
        label = "logs de servidor")
t_ajuste = np.linspace(0,max(tiempos_falla),1000)
f_ajuste = weibull_min.pdf(t_ajuste,k_estimado, scale = lamda_estimado)
ax.plot(t_ajuste, f_ajuste, "r-", lw = 3, 
        label = f"ajuste Weibull ( k = {k_estimado:.2f})")
ax.set_title("ajuste de densidad de probabilidad de tiempos de falla")
ax.set_xlabel="Tiempos de falla en horas"
ax.set_ylabel("densidad")
ax.legend
plt.show()