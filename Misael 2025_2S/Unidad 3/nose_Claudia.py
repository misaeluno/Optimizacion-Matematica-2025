# Cargar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

# Configuraciones iniciales
np.random.seed(2025)

# Parámetros del problema
n_puntos = 25

# Generación de coordenadas
puntos = np.random.uniform(low=0, high=100, size=(n_puntos, 2))
print("Primeros 5 puntos de soldadura:\n", puntos[:5])

# Visualización inicial de los puntos
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(puntos[:, 0], puntos[:, 1], color="red", marker="o", s=100, zorder=3)
for i in range(n_puntos):
    ax.annotate(str(i), (puntos[i, 0], puntos[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_title("Distribución de puntos de soldadura en la PCB")
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.grid(True, alpha=0.25)
plt.show()

# Definición de la función objetivo
def distancia_total(ruta, coordenadas):
    distancia = 0.0
    n_puntos = len(ruta)
    for i in range(n_puntos):
        id_actual = ruta[i]
        id_siguiente = ruta[(i + 1) % n_puntos]
        p1 = coordenadas[id_actual]
        p2 = coordenadas[id_siguiente]
        d = np.sqrt(np.sum((p1 - p2)**2))
        distancia += d
    return distancia

# Calcular distancia de una ruta aleatoria inicial
ruta_inicial = np.arange(n_puntos)
np.random.shuffle(ruta_inicial)
distancia_inicial = distancia_total(ruta_inicial, puntos)
print(f"Distancia (aleatoria): {distancia_inicial:.2f}")

# ALGORITMO DEL VECINO MÁS CERCANO (CORREGIDO)
# ---------------------------------------------
def vecino_mas_cercano(puntos, inicio=0):
    n = len(puntos)
    visitados = [False] * n
    ruta = [inicio]
    visitados[inicio] = True
    punto_actual = inicio
    
    # Visitar todos los puntos restantes
    for _ in range(n - 1):
        distancia_minima = float('inf')
        punto_mas_cercano = None
        
        # Buscar el punto no visitado más cercano
        for i in range(n):
            if not visitados[i]:
                # Calcular distancia euclidiana
                dist = np.sqrt(np.sum((puntos[punto_actual] - puntos[i])**2))
                
                if dist < distancia_minima:
                    distancia_minima = dist
                    punto_mas_cercano = i
        
        # Agregar el punto más cercano a la ruta
        ruta.append(punto_mas_cercano)
        visitados[punto_mas_cercano] = True
        punto_actual = punto_mas_cercano
    
    return ruta

# Ejecutar el algoritmo
ruta_optima = vecino_mas_cercano(puntos, inicio=0)
distancia_optima = distancia_total(ruta_optima, puntos)

print(f"\nRuta óptima (vecino más cercano): {ruta_optima}")
print(f"Distancia optimizada: {distancia_optima:.2f}")
print(f"Mejora: {distancia_inicial - distancia_optima:.2f} ({((distancia_inicial - distancia_optima) / distancia_inicial * 100):.1f}%)")

# Visualización de la ruta optimizada
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Ruta aleatoria
ax1.scatter(puntos[:, 0], puntos[:, 1], color="red", marker="o", s=100, zorder=3)
for i in range(n_puntos):
    id_actual = ruta_inicial[i]
    id_siguiente = ruta_inicial[(i + 1) % n_puntos]
    ax1.plot([puntos[id_actual, 0], puntos[id_siguiente, 0]], 
             [puntos[id_actual, 1], puntos[id_siguiente, 1]], 
             'b-', alpha=0.5, linewidth=1)
ax1.scatter(puntos[ruta_inicial[0], 0], puntos[ruta_inicial[0], 1], 
            color="green", marker="s", s=200, zorder=4, label="Inicio")
ax1.set_title(f"Ruta Aleatoria\nDistancia: {distancia_inicial:.2f}")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.grid(True, alpha=0.25)
ax1.legend()

# Ruta optimizada
ax2.scatter(puntos[:, 0], puntos[:, 1], color="red", marker="o", s=100, zorder=3)
for i in range(n_puntos):
    id_actual = ruta_optima[i]
    id_siguiente = ruta_optima[(i + 1) % n_puntos]
    ax2.plot([puntos[id_actual, 0], puntos[id_siguiente, 0]], 
             [puntos[id_actual, 1], puntos[id_siguiente, 1]], 
             'g-', alpha=0.7, linewidth=2)
ax2.scatter(puntos[ruta_optima[0], 0], puntos[ruta_optima[0], 1], 
            color="green", marker="s", s=200, zorder=4, label="Inicio")
ax2.set_title(f"Ruta Optimizada (Vecino Más Cercano)\nDistancia: {distancia_optima:.2f}")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.grid(True, alpha=0.25)
ax2.legend()

plt.tight_layout()
plt.show()

# Probar con diferentes puntos de inicio para mejorar resultado
print("\n--- Probando diferentes puntos de inicio ---")
mejor_ruta = None
mejor_distancia = float('inf')
mejor_inicio = 0

for inicio in range(n_puntos):
    ruta = vecino_mas_cercano(puntos, inicio)
    dist = distancia_total(ruta, puntos)
    if dist < mejor_distancia:
        mejor_distancia = dist
        mejor_ruta = ruta
        mejor_inicio = inicio

print(f"Mejor punto de inicio: {mejor_inicio}")
print(f"Mejor distancia encontrada: {mejor_distancia:.2f}")
print(f"Mejora adicional: {distancia_optima - mejor_distancia:.2f}")