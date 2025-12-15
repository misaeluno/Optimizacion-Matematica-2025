# Datos simulados pregunta 2
#-------------------------
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(2025)
# Variable independiente (metros)
#Distancia       = x
distancia = np.linspace(1, 20, 50)
# Variable dependiente (lectura sensor)
#lesctura_sensor = y
lectura_sensor = (50 / (distancia ** 1.5)) + 2 + np.random.normal(0, 0.5, 50)
datos_pregunta_2 = pd.DataFrame({"Distancia": distancia,
"Lectura_Sensor": lectura_sensor})
print(datos_pregunta_2.head())

def div(beta, distancia):
    x = np.array(distancia)
    h = (beta[0]/x**beta[1]) + beta[2]
    return h

def F(beta, distancia, lectura_sensor):
    y = np.array(lectura_sensor)
    x = np.array(distancia)
    h = div(beta, x)
    f = y - h
    f_fin = np.sum(f**2)
    return f_fin

# Datos iniciales
resultados = []
n_experimentos = 3
for i in range(n_experimentos):
    # Generar inicialización aleatoria
    beta = np.array([
        np.random.uniform(10, 100),    # β₀
        np.random.uniform(0.5, 3.0),   # β₁
        np.random.uniform(-5, 10)      # β₂
    ])
    
   # Llamada a la función de optimización
    resultado = minimize(
        fun=F,                                    # Función a minimizar
        x0=beta,                                  # Parámetros iniciales
        args=(distancia, lectura_sensor),         # Argumentos adicionales
        method='BFGS',                            # Método de optimización
        options={'disp': True})                    # Mostrar progreso 
    
    # Guardar resultados
    resultados.append({
        '/n experimento': i + 1,
        '/n beta0_inicial': beta[0],
        '/n beta1_inicial': beta[1],
        '/n beta2_inicial': beta[2],
        '/n beta0_optimo': resultado.x[0],
        '/n beta1_optimo': resultado.x[1],
        '/n beta2_optimo': resultado.x[2],
        '/n costo_final': resultado.fun,
        '/n exito': resultado.success,
        '/n n_iteraciones': resultado.nit,
        '/n n_evaluaciones': resultado.nfev
    })


print(resultados)