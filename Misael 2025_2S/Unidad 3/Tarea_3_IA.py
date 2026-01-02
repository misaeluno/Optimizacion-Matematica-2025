# Datos simulados pregunta 3
#-------------------------
import numpy as np
import matplotlib.pyplot as plt

# Semilla
np.random.seed(2025)

# S = servicios
n_servicios = 50

# K = servidores
n_servidores = 4

# Conjunto de cargas de los servicios
cargas_cpu = np.random.randint(1, 20, n_servicios)

print("=" * 70)
print("PROBLEMA: BALANCEO DE CARGA DE MICROSERVICIOS")
print("=" * 70)
print(f"Número de servicios: {n_servicios}")
print(f"Número de servidores: {n_servidores}")
print(f"Carga total: {np.sum(cargas_cpu)} unidades")
print(f"Carga ideal por servidor: {np.sum(cargas_cpu) / n_servidores:.2f}")
print()

# ===================================================================
# FUNCIONES BASE (TU CÓDIGO)
# ===================================================================

def Lk(cargas_cpu, n_servidores, A):
    """
    Calcula la carga total de cada servidor.
    """
    s = np.array(cargas_cpu)
    k = n_servidores
    lk = np.zeros(k)
    
    # Sumar la carga de cada servicio a su servidor asignado
    for i in range(len(A)):
        save = A[i]      # Servidor del servicio i
        joker = s[i]     # Carga del servicio i
        lk[save] += joker
    
    return lk

def f_objetivo(cargas_cpu, n_servidores, A):
    """
    Función objetivo: Desviación estándar de las cargas.
    """
    s = np.array(cargas_cpu)
    k = n_servidores
    lk = Lk(s, k, A)
    
    # Calcular media de cargas
    carga_media = np.mean(lk)
    
    # Calcular desviación estándar
    lk_diferencia = lk - carga_media
    lk_cuadrado = lk_diferencia**2
    f = (1/n_servidores) * np.sum(lk_cuadrado)
    f = np.sqrt(f)
    
    return f

# ===================================================================
# SIMULATED ANNEALING CORREGIDO
# ===================================================================

def generar_vecino(A_actual, n_servidores):
    """
    ✅ CORRECCIÓN: Genera vecino moviendo UN servicio a OTRO servidor.
    
    NO es un problema de rutas ni distancias.
    Es un problema de ASIGNACIÓN de servicios a servidores.
    """
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

def simulated_annealing(cargas_cpu, n_servidores, 
                       t_inicial=1000, 
                       t_final=0.1, 
                       alpha=0.95, 
                       max_iter=10000):
    """
    ✅ CORRECCIÓN: Simulated Annealing para BALANCEO DE CARGA.
    
    NO es el problema del viajante (TSP).
    Minimiza la desviación estándar de cargas entre servidores.
    """
    # Asignación inicial aleatoria
    A_actual = np.random.randint(0, n_servidores, n_servicios)
    f_actual = f_objetivo(cargas_cpu, n_servidores, A_actual)
    
    # Mejor solución
    A_mejor = A_actual.copy()
    f_mejor = f_actual
    
    # Variables de control
    temperatura = t_inicial
    iteraciones = 0
    
    # Historial
    historial_costo = []
    historial_mejor = []
    historial_temp = []
    
    print("Ejecutando Simulated Annealing...")
    print(f"Desviación inicial: {f_actual:.4f}")
    print()
    
    while temperatura > t_final and iteraciones < max_iter:
        # Generar vecino
        A_nuevo = generar_vecino(A_actual, n_servidores)
        f_nuevo = f_objetivo(cargas_cpu, n_servidores, A_nuevo)
        
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
        historial_mejor.append(f_mejor)
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
    
    return A_mejor, f_mejor, iteraciones, historial_costo, historial_mejor, historial_temp

# ===================================================================
# EJECUTAR SIMULATED ANNEALING
# ===================================================================

A_mejor, f_mejor, iteraciones, hist_costo, hist_mejor, hist_temp = simulated_annealing(
    cargas_cpu, 
    n_servidores,
    t_inicial=1000,
    t_final=0.1,
    alpha=0.95,
    max_iter=10000
)

# ===================================================================
# ANÁLISIS DE RESULTADOS
# ===================================================================

print("=" * 70)
print("RESULTADOS FINALES")
print("=" * 70)

# Comparar con otras estrategias
A_aleatorio = np.random.randint(0, n_servidores, n_servicios)
f_aleatorio = f_objetivo(cargas_cpu, n_servidores, A_aleatorio)
cargas_aleatorio = Lk(cargas_cpu, n_servidores, A_aleatorio)

A_greedy = np.arange(n_servicios) % n_servidores
f_greedy = f_objetivo(cargas_cpu, n_servidores, A_greedy)
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

print("\nMejoras logradas:")
mejora_aleatorio = ((f_aleatorio - f_mejor) / f_aleatorio) * 100
mejora_greedy = ((f_greedy - f_mejor) / f_greedy) * 100
print(f"   vs Aleatorio: {mejora_aleatorio:.2f}%")
print(f"   vs Greedy:    {mejora_greedy:.2f}%")

print("\nMétricas de balance:")
print(f"   Carga mínima:  {np.min(cargas_sa):.0f}")
print(f"   Carga máxima:  {np.max(cargas_sa):.0f}")
print(f"   Diferencia:    {np.max(cargas_sa) - np.min(cargas_sa):.0f}")
print(f"   Carga ideal:   {np.sum(cargas_cpu) / n_servidores:.2f}")
print("=" * 70)
print()

# ===================================================================
# VISUALIZACIONES
# ===================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Convergencia
ax1 = axes[0, 0]
ax1.plot(hist_costo, color='blue', alpha=0.5, linewidth=1, label='Costo actual')
ax1.plot(hist_mejor, color='red', linewidth=2, label='Mejor costo')
ax1.set_xlabel('Iteración', fontsize=10)
ax1.set_ylabel('Desviación Estándar', fontsize=10)
ax1.set_title('Convergencia del Algoritmo', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. Temperatura
ax2 = axes[0, 1]
ax2.plot(hist_temp, color='orange', linewidth=2)
ax2.set_xlabel('Iteración', fontsize=10)
ax2.set_ylabel('Temperatura', fontsize=10)
ax2.set_title('Esquema de Enfriamiento', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Comparación de desviaciones
ax3 = axes[0, 2]
estrategias = ['Aleatorio', 'Greedy', 'SA']
desviaciones = [f_aleatorio, f_greedy, f_mejor]
colores = ['red', 'orange', 'green']
bars = ax3.bar(estrategias, desviaciones, color=colores, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, desviaciones):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.2f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel('Desviación Estándar', fontsize=10)
ax3.set_title('Comparación de Estrategias', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Cargas - Aleatorio
ax4 = axes[1, 0]
servidores = [f'S{i}' for i in range(n_servidores)]
ax4.bar(servidores, cargas_aleatorio, color='red', alpha=0.7, edgecolor='black')
ax4.axhline(np.mean(cargas_aleatorio), color='blue', linestyle='--', linewidth=2)
ax4.set_ylabel('Carga (CPU)', fontsize=10)
ax4.set_title(f'Aleatorio (σ={f_aleatorio:.2f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. Cargas - Greedy
ax5 = axes[1, 1]
ax5.bar(servidores, cargas_greedy, color='orange', alpha=0.7, edgecolor='black')
ax5.axhline(np.mean(cargas_greedy), color='blue', linestyle='--', linewidth=2)
ax5.set_ylabel('Carga (CPU)', fontsize=10)
ax5.set_title(f'Greedy (σ={f_greedy:.2f})', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# 6. Cargas - SA
ax6 = axes[1, 2]
ax6.bar(servidores, cargas_sa, color='green', alpha=0.7, edgecolor='black')
ax6.axhline(np.mean(cargas_sa), color='blue', linestyle='--', linewidth=2)
ax6.set_ylabel('Carga (CPU)', fontsize=10)
ax6.set_title(f'SA (σ={f_mejor:.2f})', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('Balanceo de Carga con Simulated Annealing', 
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()

# ===================================================================
# VISUALIZACIÓN ADICIONAL: DISTRIBUCIÓN DE SERVICIOS
# ===================================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Crear gráfico de barras apiladas mostrando qué servicios van a cada servidor
for k in range(n_servidores):
    servicios_en_k = np.where(A_mejor == k)[0]
    cargas_en_k = cargas_cpu[servicios_en_k]
    
    left = 0
    for servicio, carga in zip(servicios_en_k, cargas_en_k):
        ax.barh(k, carga, left=left, alpha=0.8, edgecolor='black', linewidth=0.5)
        if carga > 5:  # Solo mostrar etiqueta si la carga es visible
            ax.text(left + carga/2, k, f'{servicio}', 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        left += carga

ax.set_yticks(range(n_servidores))
ax.set_yticklabels([f'Servidor {i}' for i in range(n_servidores)])
ax.set_xlabel('Carga Total (CPU)', fontsize=11)
ax.set_title('Distribución de Servicios por Servidor', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("\n✓ Ejecución completada exitosamente!")