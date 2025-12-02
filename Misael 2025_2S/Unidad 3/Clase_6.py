# Cargar librerías necesarias
# ---------------------------
import numpy as np
import matplotlib.pyplot as plt
# Configuraciones iniciales
# -------------------------

# Conjunto de datos (simulados)
# -----------------------------
np.random.seed(2025)

# Parámetros del problema
n_puntos = 25

# Generación de coordenadas
puntos = np.random.uniform(low = 0, high = 100, size = (n_puntos, 2))
print("Primeros 5 puntos de soldadura:\n", puntos[:5])

# Visualización inicial de los puntos
fig, ax = plt.subplots(figsize = (8, 8))
ax.scatter(puntos[:, 0], puntos[:, 1], color = "red", marker = "o")
ax.set_title("Distribución de puntos de soldadura en la PCB")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.grid(True, alpha = 0.25)
plt.show()

# Definición del función objetivo
# -------------------------------
def distancia_total(ruta, coordenadas):
    distancia = 0.0
    n_puntos = len(ruta)
    for i in range(n_puntos):
        # Índice punto actual
        id_actual = ruta[i]
        # Índice siguiente punto
        id_siguiente = ruta[(i + 1) % n_puntos]
        # Coordenadas de los puntos
        p1 = coordenadas[id_actual]
        p2 = coordenadas[id_siguiente]
        # Distancia Euclidiana
        d = np.sqrt(np.sum((p1 - p2)**2))
        distancia = distancia + d
    return distancia

# Calcular distancia de una ruta aleatoria inicial
ruta_inicial = np.arange(n_puntos) # [0, 1, 2, ..., 24]
np.random.shuffle(ruta_inicial)
distancia_inicial = distancia_total(ruta_inicial, puntos)
print(f"Distancia (aleatoria): {distancia_inicial:.2f}")
