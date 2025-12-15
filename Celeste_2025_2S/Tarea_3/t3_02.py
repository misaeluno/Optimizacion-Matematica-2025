import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def main():
    # Semilla para reproducibilidad
    np.random.seed(2025)
    r = np.linspace(1, 20, 50)
    l = (50 / (r**1.5)) + 2 + np.random.normal(0, 0.5, 50)
    d = pd.DataFrame({"razstojanje": r, "read": l})
    print(d.head())

    # Funciones
    def h(theta):
        return (theta[0] / (r ** theta[1])) + theta[2]

    def F(theta):
        return np.sum((l - h(theta)) ** 2)

    # Gradiente
    def gradient(theta):
        h_val = h(theta)
        residual = l - h_val

        # ∂F/∂θ₀ = Σ 2*(l - h) * (1 / r^θ₁)
        dtheta0 = np.sum(2 * residual * (1 / (r ** theta[1])))

        # ∂F/∂θ₁ = Σ -2*(l - h) * θ₀ * ln(r) / r^θ₁
        dtheta1 = np.sum(-2 * residual * theta[0] * np.log(r) / (r ** theta[1]))

        # ∂F/∂θ₂ = Σ 2*(l - h)
        dtheta2 = np.sum(2 * residual)

        return np.array([dtheta0, dtheta1, dtheta2])

    # Valor inicial mejorado (cerca de los valores verdaderos)
    theta_inicial = np.array([40.0, 1.2, 3.0])  # Cerca de [50, 1.5, 2]

    # Optimización
    resultado = minimize(
        F,
        theta_inicial,
        jac=gradient,
        method="BFGS",
        options={"maxiter": 1000, "disp": True, "gtol": 1e-8},
    )

    theta_opt = resultado.x
    print("\nResultados de la optimización:")
    print(f"θ optimizado: {theta_opt}")
    print(f"θ verdadero: [50.0, 1.5, 2.0]")
    print(f"Error cuadrático mínimo: {resultado.fun:.6f}")

    # Visualización
    plt.figure(figsize=(12, 6))

    # Datos
    plt.scatter(r, l, alpha=0.7, label="Datos observados", color="blue")

    # Curva ajustada
    r_suave = np.linspace(1, 20, 200)
    l_ajustado = (theta_opt[0] / (r_suave ** theta_opt[1])) + theta_opt[2]
    plt.plot(
        r_suave,
        l_ajustado,
        "r-",
        linewidth=2,
        label=f"Ajuste: {theta_opt[0]:.2f}/r^{theta_opt[1]:.2f} + {theta_opt[2]:.2f}",
    )

    # Curva verdadera (sin ruido)
    l_verdadero = (50 / (r_suave**1.5)) + 2
    plt.plot(r_suave, l_verdadero, "g--", linewidth=2, label="Verdadero: 50/r^1.5 + 2")

    plt.xlabel("Distancia (r)")
    plt.ylabel("Lectura (l)")
    plt.title("Ajuste no lineal por mínimos cuadrados")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()
