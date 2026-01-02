# Datos simulados pregunta 3
#-------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mate
import scipy.special

# semilla
np.random.seed(2025)
# S = servicios
n_servicios = 50
# K = servidor
n_servidores = 4
#conjunto de carga de los servicios
cargas_cpu = np.random.randint(1, 20, n_servicios)
# Asignacion
A = np.random.randint(0, n_servidores, n_servicios)
t_inicial=1000 
t_final=0.1 
alpha=0.3 
max_iter=1000


def Lk (cargas_cpu, n_servidores,A):

    s = np.array(cargas_cpu)
    k = n_servidores
    lk = np.zeros(k)
    
    # Sumar la carga de cada servicio a su servidor asignado
    for i in range(len(A)):
        save = A[i]  # Servidor del servicio i
        joker = s[i]      # Carga del servicio i
        lk[save] += joker

    return lk

def f_objetico(cargas_cpu, n_servidores,A):
    
    s = np.array(cargas_cpu)
    k = n_servidores

    lk = Lk(s, k, A)
    
    # Calcular media de cargas
    carga_media = np.mean(lk)
    
    lk_diferencia= lk - carga_media
    lk_cuadrado = lk_diferencia**2
    f = (1/n_servidores) * np.sum(lk_cuadrado)
    f = np.sqrt(f)
    print(f)
    return f

def generar_vecino(A_actual, n_servidores):

    A_vecino = A_actual.copy()
    # Seleccionar un servicio aleatorio
    servicio = np.random.randint(0, len(A_vecino))
    
    # Seleccionar un servidor diferente al actual
    servidor_actual = A_vecino[servicio]
    servidores_disponibles = [s for s in range(n_servidores) if s != servidor_actual]
    nuevo_servidor = np.random.choice(servidores_disponibles)
    
    # Mover el servicio
    A_vecino[servicio] = nuevo_servidor
    
    return A_vecino

def simulated_annealing(cargas_cpu, n_servidores, t_inicial, t_final, alpha, max_iter,A):
 
    # Asignación inicial aleatoria
    A_actual = A.copy()
    f_actual = f_objetico(cargas_cpu, n_servidores, A_actual)
    
    # Mejor solución
    A_mejor = A_actual.copy()
    f_mejor = f_actual
    
    # Variables de control
    temperatura = t_inicial
    iteraciones = 0
    
    # Historial
    historial_costo = []
    historial_temp = []
    while temperatura > t_final and iteraciones < max_iter:
        # Generar vecino
        A_nuevo = generar_vecino(A_actual, n_servidores)
        f_nuevo = f_objetico(cargas_cpu, n_servidores, A_nuevo)
        
        # Calcular delta
        delta = f_nuevo - f_actual
        
        # Criterio de aceptación
        if delta < 0:
            # Mejor solución: aceptar
            A_actual = A_nuevo
            f_actual = f_nuevo
            
            if f_actual < f_mejor:
                A_mejor = A_actual.copy()
                f_mejor = f_actual
        else:
            # Peor solución: aceptar con probabilidad
            probabilidad = np.exp(-delta / temperatura)
            if np.random.rand() < probabilidad:
                A_actual = A_nuevo
                f_actual = f_nuevo
        
        # Registrar historial
        historial_costo.append(f_actual)
        historial_temp.append(temperatura)
        # Enfriar
        temperatura = alpha * temperatura
        iteraciones += 1
        
        # Mostrar progreso cada 1000 iteraciones
        if iteraciones % 1000 == 0:
            print(f"Iter {iteraciones:5d} | T={temperatura:7.2f} | "
                  f"Actual={f_actual:7.4f} | Mejor={f_mejor:7.4f}")
    
    print()
    print(f"✓ Convergencia alcanzada!")
    print(f"Iteraciones totales: {iteraciones}")
    print(f"Desviación final: {f_mejor:.4f}")
    print()
    
    return A_mejor, f_mejor, iteraciones, historial_costo, historial_temp

A_mejor, f_mejor, iteraciones, hist_costo, hist_temp = simulated_annealing(cargas_cpu, n_servidores, t_inicial, 
                                                                            t_final,alpha, max_iter, A)

# Comparar con otras estrategias
A_aleatorio = np.random.randint(0, n_servidores, n_servicios)
f_aleatorio = f_objetico(cargas_cpu, n_servidores, A_aleatorio)
cargas_aleatorio = Lk(cargas_cpu, n_servidores, A_aleatorio)

A_greedy = np.arange(n_servicios) % n_servidores
f_greedy = f_objetico(cargas_cpu, n_servidores, A_greedy)
cargas_greedy = Lk(cargas_cpu, n_servidores, A_greedy)

cargas_sa = Lk(cargas_cpu, n_servidores, A_mejor)

print("\n1. ASIGNACIÓN ALEATORIA:")
print(f"   Desviación: {f_aleatorio:.4f}")
print(f"   Cargas: {cargas_aleatorio}")

print("\n2. ASIGNACIÓN GREEDY (Round Robin):")
print(f"   Desviación: {f_greedy:.4f}")
print(f"   Cargas: {cargas_greedy}")

print("\n3. SIMULATED ANNEALING:")
print(f"   Desviación: {f_mejor:.4f}")
print(f"   Cargas: {cargas_sa}")


fig, ax1 = plt.subplots(figsize=(12, 7))

# Plotear temperatura
ax1.plot(hist_temp, color='darkorange', linewidth=2.5, label='Temperatura')
ax1.set_xlabel('Iteración', fontsize=13, fontweight='bold')
ax1.set_ylabel('Temperatura', fontsize=13, fontweight='bold', color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')
ax1.grid(True, alpha=0.3, linestyle='--')

# Título y leyendas
ax1.set_title('Esquema de Enfriamiento\n' + 
              f'T_inicial={t_inicial}, T_final={t_final}, alpha={alpha}',
              fontsize=14, fontweight='bold', pad=15)

# Combinar leyendas
lines1, labels1 = ax1.get_legend_handles_labels()

# Añadir anotaciones
ax1.annotate(f'Inicio\nT = {hist_temp[0]:.0f}', 
            xy=(0, hist_temp[0]), xytext=(iteraciones*0.15, hist_temp[0]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold', color='red')

ax1.annotate(f'Final\nT = {hist_temp[-1]:.2f}', 
            xy=(len(hist_temp)-1, hist_temp[-1]), 
            xytext=(iteraciones*0.7, hist_temp[-1] + 100),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=10, fontweight='bold', color='green')

plt.tight_layout()
plt.show()
