import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # Semilla para reproducibilidad
    np.random.seed(2025)
    m = 100
    u = np.random.uniform(10, 100, m)
    l = np.exp(0.5 + 0.04 * u)  # True model: λ = exp(0.5 + 0.04*u)
    p = np.random.poisson(l)
    d = pd.DataFrame({"Usuarios": u, "Peticiones": p})
    print(d.head())
    
    # Parámetros gradiente
    alpha = 0.000000119  # Tasa de aprendizaje
    epsilon = 1e-6  # Tolerancia para convergencia
    x = np.column_stack((np.ones(m), u.reshape(-1, 1)))  # Matriz de diseño con intercepto
    beta = np.array([0.0, 0.0])  # Inicialización
    
    # Funciones
    def lambda_func(beta): 
        return np.exp(x @ beta)  # Poisson rate: λ = exp(β₀ + β₁*u)
    
    # J(β) = -log-likelihood (negativa para minimizar)
    # log-likelihood Poisson: Σ[y_i*log(λ_i) - λ_i]
    def J(beta): 
        return -np.sum(p * np.log(lambda_func(beta)) - lambda_func(beta))
    
    # Gradiente: ∇J(β) = Xᵀ(λ - y)
    def gradient(beta): 
        return x.T @ (lambda_func(beta) - p)
    
    # Actualización gradiente descendente
    def gradient_update(beta, alpha): 
        return beta - alpha * gradient(beta)
    
    phi = [J(beta), J(beta)]
    
    print(f"Valores verdaderos: β₀ = 0.5, β₁ = 0.04")
    print(f"Inicial: β = {beta}, J(β) = {phi[0]:.4f}")
    
    for i in range(100000):
        beta = gradient_update(beta, alpha)
        phi[0], phi[1] = J(beta), phi[0]
        
        if i % 1000 == 0:
            print(f"Iteración {i}: β = [{beta[0]:.6f}, {beta[1]:.6f}], J(β) = {phi[0]:.6f}")
            print(f"  Gradiente: ∇J = [{gradient(beta)[0]:.6e}, {gradient(beta)[1]:.6e}]")
        
        # Criterios de convergencia
        if np.linalg.norm(gradient(beta)) < epsilon:  # Norma del gradiente cercana a cero
            print(f"\nConvergencia alcanzada en {i} iteraciones")
            print(f"Norma del gradiente: {np.linalg.norm(gradient(beta)):.6e}")
            break
            
        if abs(phi[0] - phi[1]) < epsilon:  # Cambio mínimo en función objetivo
            print(f"\nConvergencia por cambio mínimo en {i} iteraciones")
            break
    
    print(f"\nResultado final:")
    print(f"β estimado = [{beta[0]:.6f}, {beta[1]:.6f}]")
    print(f"β verdadero = [0.500000, 0.040000]")
    print(f"Error absoluto = [{abs(beta[0]-0.5):.6f}, {abs(beta[1]-0.04):.6f}]")
    print(f"J(β) final = {phi[0]:.6f}")
    
    # 3. Predecir la tasa de peticiones para una carga de usuarios inédita
    print("\n" + "="*50)
    print("3. PREDICCIÓN PARA USUARIOS INÉDITOS:")
    print("="*50)
    
    # Generar nuevos usuarios (fuera del rango de entrenamiento)
    nuevos_usuarios = np.array([5, 50, 105, 150])  # Incluye valores fuera de [10, 100]
    
    for u_nuevo in nuevos_usuarios:
        # Crear vector de características con intercepto
        x_nuevo = np.array([1.0, u_nuevo])
        
        # Predecir tasa λ (poisson rate)
        lambda_pred = np.exp(beta @ x_nuevo)
        
        # Para Poisson, la media es λ
        prediccion_media = lambda_pred
        
        print(f"\nPara {u_nuevo} usuarios:")
        print(f"  Tasa λ predicha: {lambda_pred:.4f}")
        print(f"  Número esperado de peticiones: {prediccion_media:.2f}")
        
        # También podemos calcular un intervalo de confianza aproximado
        # Para Poisson, la desviación estándar es sqrt(λ)
        std_pred = (lambda_pred ** 0.5)
        print(f"  Intervalo aproximado 95%: [{max(0, lambda_pred - 1.96*std_pred):.2f}, "
              f"{lambda_pred + 1.96*std_pred:.2f}]")
    
    # Visualización
    print("\n" + "="*50)
    print("COMPARACIÓN VISUAL:")
    print("="*50)
    
    # Usuarios ordenados para visualización
    u_ordenado = np.array(sorted(u))
    
    # Predicciones para usuarios de entrenamiento
    x_entrenamiento = np.column_stack((np.ones(len(u_ordenado)), u_ordenado))
    lambda_entrenamiento = np.exp(x_entrenamiento @ beta)
    
    # Gráfico
    plt.figure(figsize=(12, 6))
    
    # Datos reales
    plt.scatter(u, p, alpha=0.5, label='Datos observados', color='blue')
    
    # Curva ajustada
    plt.plot(u_ordenado, lambda_entrenamiento, 'r-', linewidth=2, 
            label=f'Modelo: λ = exp({beta[0]:.3f} + {beta[1]:.3f}*u)')
    
    # Curva verdadera (si la conocemos)
    lambda_verdadero = np.exp(0.5 + 0.04 * u_ordenado)
    plt.plot(u_ordenado, lambda_verdadero, 'g--', linewidth=2, 
            label='Verdadero: λ = exp(0.5 + 0.04*u)')
    
    plt.xlabel('Número de usuarios (u)')
    plt.ylabel('Tasa de peticiones (λ)')
    plt.title('Regresión de Poisson: Ajuste del modelo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mostrar predicciones para usuarios inéditos
    for u_nuevo in nuevos_usuarios:
        lambda_nuevo = np.exp(beta @ np.array([1.0, u_nuevo]))
        plt.scatter(u_nuevo, lambda_nuevo, s=100, color='purple', marker='*', 
                  label=f'Predicción {u_nuevo} usuarios' if u_nuevo == nuevos_usuarios[0] else "")
        plt.text(u_nuevo, lambda_nuevo, f'  {lambda_nuevo:.1f}', verticalalignment='bottom')
    
    plt.show()


if __name__ == "__main__":
    main()
