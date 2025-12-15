from typing import Any, Callable, Iterator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_neighbor(a, m):
    """Genera un vecino moviendo un elemento a otro grupo"""
    _a = a.copy()
    i = np.random.randint(len(_a))  # Selecciona un elemento aleatorio
    _s = _a[i]
    
    # Lista de grupos posibles (excluyendo el actual)
    possible_groups = list(range(m))
    if _s in possible_groups:
        possible_groups.remove(_s)
    
    _a[i] = np.random.choice(possible_groups)  # Asigna a un grupo diferente
    return _a


class SimulatedAnnealing(Iterator):
    def __init__(
        self, f: Callable, temperature: float = 100, alpha: float = 0.95, m: float = 5
    ) -> None:
        self.f = f
        self.best = ([], float('inf'))
        self.current = ([], float('inf'))
        self.index = 0
        self.temperature = temperature
        self.alpha = alpha
        self.m = m
        
        # Historial para análisis
        self.history = {
            'f_values': [],
            'temperatures': [],
            'accepted_worse': [],
            'best_values': []
        }
        self.stats = {
            'total_iterations': 0,
            'accepted_better': 0,
            'accepted_worse': 0,
            'rejected': 0
        }

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.best[0]
        else:
            return self.best[1]

    def __len__(self):
        return 2

    def __next__(self):
        if self.index < len(self):
            self.index += 1
            return self.best[0]
        raise StopIteration

    def __call__(self, e: np.ndarray, iterations: int = 10000) -> np.ndarray:
        self.best = (e.copy(), self.f(e))
        self.current = (e.copy(), self.f(e))
        
        # Inicializar historial
        self.history['f_values'].append(self.current[1])
        self.history['temperatures'].append(self.temperature)
        self.history['best_values'].append(self.best[1])
        self.history['accepted_worse'].append(0)
        
        print(f"Inicial: f(x) = {self.current[1]:.4f}")

        for iteration in range(iterations):
            # Generar vecino
            _e = generate_neighbor(self.current[0], self.m)
            _f = self.f(_e)
            
            # Cambio en la función objetivo
            delta = _f - self.current[1]
            
            # Criterio de aceptación
            if delta < 0:
                # Mejor solución → siempre aceptar
                self.current = (_e.copy(), _f)
                self.stats['accepted_better'] += 1
                accepted = True
            else:
                # Solución peor → aceptar con probabilidad
                p = np.exp(-delta / self.temperature)
                if np.random.rand() < p:
                    self.current = (_e.copy(), _f)
                    self.stats['accepted_worse'] += 1
                    accepted = True
                else:
                    self.stats['rejected'] += 1
                    accepted = False
            
            # Actualizar mejor solución global
            if self.current[1] < self.best[1]:
                self.best = (self.current[0].copy(), self.current[1])
            
            # Registrar en historial
            self.history['f_values'].append(self.current[1])
            self.history['best_values'].append(self.best[1])
            self.history['temperatures'].append(self.temperature)
            self.history['accepted_worse'].append(1 if delta >= 0 and accepted else 0)
            
            # Enfriar temperatura
            self.temperature *= self.alpha
            
            # Mostrar progreso
            if iteration % 1000 == 0:
                print(f"Iteración {iteration}: f = {self.current[1]:.4f}, "
                      f"mejor = {self.best[1]:.4f}, T = {self.temperature:.4f}")
        
        self.stats['total_iterations'] = iterations
        return self[0]


def main():
    # Semilla para reproducibilidad
    np.random.seed(2025)
    s = 50
    m = 4
    
    # Generar datos (valores que sumar en cada grupo)
    c = np.random.randint(1, 20, s)
    d = pd.DataFrame({"valor": c})
    print(d.head())
    
    # Función objetivo: varianza de las sumas de los grupos
    def f(a):
        group_sums = np.zeros(m)  # Inicializar suma por grupo
        
        for i, j in enumerate(a):
            group_sums[j] += c[i]  # Sumar valor al grupo correspondiente
        
        return np.var(group_sums)  # Varianza de las sumas
    
    def get_group_sums(a):
        """Obtiene las sumas por grupo"""
        group_sums = np.zeros(m)
        for i, j in enumerate(a):
            group_sums[j] += c[i]
        return group_sums
    
    # Solución inicial aleatoria
    a = np.random.randint(0, m, s)
    initial_sums = get_group_sums(a)
    
    print(f"\n{'='*60}")
    print("PROBLEMA DE ASIGNACIÓN PARA MINIMIZAR VARIANZA")
    print(f"{'='*60}")
    print(f"Elementos: {s}")
    print(f"Grupos: {m}")
    print(f"Valores total a distribuir: {sum(c)}")
    print(f"\nSolución inicial:")
    print(f"x = {a}")
    print(f"Sumas por grupo: {initial_sums}")
    print(f"Media de sumas: {np.mean(initial_sums):.2f}")
    print(f"Varianza inicial f(x) = {f(a):.4f}")
    print(f"Desviación estándar: {np.std(initial_sums):.4f}")
    
    # Ejecutar recocido simulado
    sa = SimulatedAnnealing(f, temperature=100, alpha=0.99, m=m)
    optimal_a = sa(a, iterations=5000)
    optimal_sums = get_group_sums(optimal_a)
    
    print(f"\n{'='*60}")
    print("RESULTADOS DEL RECOCIDO SIMULADO")
    print(f"{'='*60}")
    print(f"\nSolución óptima encontrada:")
    print(f"x = {optimal_a}")
    print(f"Sumas por grupo: {optimal_sums}")
    print(f"Media de sumas: {np.mean(optimal_sums):.2f}")
    print(f"Varianza óptima f(x) = {sa.best[1]:.4f}")
    print(f"Desviación estándar: {np.std(optimal_sums):.4f}")
    
    # Estadísticas del algoritmo
    print(f"\nEstadísticas del algoritmo:")
    print(f"  Iteraciones totales: {sa.stats['total_iterations']}")
    print(f"  Soluciones mejores aceptadas: {sa.stats['accepted_better']}")
    print(f"  Soluciones peores aceptadas: {sa.stats['accepted_worse']}")
    print(f"  Soluciones rechazadas: {sa.stats['rejected']}")
    print(f"  Tasa de aceptación de peores: "
          f"{sa.stats['accepted_worse']/(sa.stats['total_iterations'])*100:.2f}%")
    
    # Análisis de equilibrio
    print(f"\n{'='*60}")
    print("ANÁLISIS DEL DESEQUILIBRIO")
    print(f"{'='*60}")
    
    ideal_sum = sum(c) / m
    print(f"\nSuma ideal por grupo: {ideal_sum:.2f}")
    
    print(f"\nDesequilibrio inicial:")
    for i in range(m):
        imbalance = (initial_sums[i] - ideal_sum) / ideal_sum * 100
        print(f"  Grupo {i}: {initial_sums[i]:.1f} ({imbalance:+.1f}%)")
    
    print(f"\nDesequilibrio final:")
    for i in range(m):
        imbalance = (optimal_sums[i] - ideal_sum) / ideal_sum * 100
        print(f"  Grupo {i}: {optimal_sums[i]:.1f} ({imbalance:+.1f}%)")
    
    # Reducción porcentual
    initial_std = np.std(initial_sums)
    final_std = np.std(optimal_sums)
    reduction = (initial_std - final_std) / initial_std * 100
    
    print(f"\nReducción del desequilibrio:")
    print(f"  Desviación estándar inicial: {initial_std:.4f}")
    print(f"  Desviación estándar final:   {final_std:.4f}")
    print(f"  Reducción: {reduction:.1f}%")
    
    # VISUALIZACIONES
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Evolución de la función objetivo
    ax1 = plt.subplot(2, 3, 1)
    iterations_range = np.arange(len(sa.history['f_values']))
    ax1.plot(iterations_range, sa.history['f_values'], 'b-', alpha=0.6, label='f(x) actual')
    ax1.plot(iterations_range, sa.history['best_values'], 'r-', linewidth=2, label='Mejor f(x)')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Varianza f(x)')
    ax1.set_title('Evolución de la Función Objetivo')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperatura vs Iteración
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogy(iterations_range, sa.history['temperatures'], 'g-')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Temperatura (log)')
    ax2.set_title('Enfriamiento (Schedule)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Aceptación de soluciones peores
    ax3 = plt.subplot(2, 3, 3)
    # Promedio móvil de aceptaciones peores
    window = 100
    worse_acceptance = sa.history['accepted_worse']
    moving_avg = [sum(worse_acceptance[max(0, i-window):i+1])/(min(window, i+1)) 
                  for i in range(len(worse_acceptance))]
    ax3.plot(iterations_range, moving_avg, 'm-')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Tasa de aceptación de peores')
    ax3.set_title('Exploración vs Explotación')
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribución inicial vs final (barras)
    ax4 = plt.subplot(2, 3, 4)
    bar_width = 0.35
    x_pos = np.arange(m)
    ax4.bar(x_pos - bar_width/2, initial_sums, bar_width, 
            label='Inicial', alpha=0.7, color='red')
    ax4.bar(x_pos + bar_width/2, optimal_sums, bar_width, 
            label='Óptimo', alpha=0.7, color='green')
    ax4.axhline(y=ideal_sum, color='k', linestyle='--', alpha=0.5, label='Ideal')
    ax4.set_xlabel('Grupo')
    ax4.set_ylabel('Suma')
    ax4.set_title('Distribución por Grupo')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([f'Grupo {i}' for i in range(m)])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Histograma de valores por grupo (inicial)
    ax5 = plt.subplot(2, 3, 5)
    colors = cm.rainbow(np.arange(m)/m)
    for group in range(m):
        group_values = [c[i] for i in range(s) if a[i] == group]
        ax5.hist(group_values, alpha=0.5, label=f'Grupo {group}', 
                color=colors[group], bins=10)
    ax5.set_xlabel('Valor')
    ax5.set_ylabel('Frecuencia')
    ax5.set_title('Distribución Inicial por Grupo')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Histograma de valores por grupo (óptimo)
    ax6 = plt.subplot(2, 3, 6)
    for group in range(m):
        group_values = [c[i] for i in range(s) if optimal_a[i] == group]
        ax6.hist(group_values, alpha=0.5, label=f'Grupo {group}', 
                color=colors[group], bins=10)
    ax6.set_xlabel('Valor')
    ax6.set_ylabel('Frecuencia')
    ax6.set_title('Distribución Óptima por Grupo')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Gráfico adicional: matriz de asignación
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Matriz de asignación inicial
    assignment_matrix_init = np.zeros((m, 20))
    for i in range(s):
        group = a[i]
        value = c[i]
        # Encontrar primera columna vacía en esta fila
        col = 0
        while assignment_matrix_init[group, col] != 0:
            col += 1
        assignment_matrix_init[group, col] = value
    
    im7 = ax7.imshow(assignment_matrix_init, cmap='YlOrRd', aspect='auto')
    ax7.set_xlabel('Posición en grupo')
    ax7.set_ylabel('Grupo')
    ax7.set_title('Asignación Inicial (matriz)')
    plt.colorbar(im7, ax=ax7, label='Valor')
    ax7.set_xticks([])
    
    # Matriz de asignación óptima
    assignment_matrix_opt = np.zeros((m, 20))
    for i in range(s):
        group = optimal_a[i]
        value = c[i]
        col = 0
        while assignment_matrix_opt[group, col] != 0:
            col += 1
        assignment_matrix_opt[group, col] = value
    
    im8 = ax8.imshow(assignment_matrix_opt, cmap='YlOrRd', aspect='auto')
    ax8.set_xlabel('Posición en grupo')
    ax8.set_ylabel('Grupo')
    ax8.set_title('Asignación Óptima (matriz)')
    plt.colorbar(im8, ax=ax8, label='Valor')
    ax8.set_xticks([])
    
    plt.tight_layout()
    plt.show()
    
    # Resumen final
    print(f"\n{'='*60}")
    print("RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"1. La varianza se redujo de {f(a):.4f} a {sa.best[1]:.4f}")
    print(f"2. La desviación estándar se redujo un {reduction:.1f}%")
    print(f"3. La suma ideal por grupo es {ideal_sum:.2f}")
    print(f"4. El máximo desequilibrio inicial: "
          f"{max(abs(s - ideal_sum) for s in initial_sums):.2f}")
    print(f"5. El máximo desequilibrio final: "
          f"{max(abs(s - ideal_sum) for s in optimal_sums):.2f}")
    
    return sa, a, optimal_a


if __name__ == "__main__":
    sa, initial, optimal = main()