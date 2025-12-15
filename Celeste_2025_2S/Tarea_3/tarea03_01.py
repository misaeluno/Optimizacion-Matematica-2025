from matplotlib import pyplot as ⱂ
from numpy import abs as Ⰰ
from numpy import array as Ⰲ
from numpy import clip as Ⱌ
from numpy import column_stack as Ⰽ
from numpy import exp as Ⰵ
from numpy import linspace as ⰾ
from numpy import log as Ⰾ
from numpy import meshgrid as ⱞ
from numpy import ones as Ⱁ
from numpy import sum as Ⱄ
from numpy.random import poisson as Ⱂ
from numpy.random import seed as ⱄ
from numpy.random import uniform as Ⱆ
from pandas import DataFrame as Ⰴ
from scipy.special import gamma as Ⰳ


def main():
    # Intro Profe
    ⱄ(2025)
    m = 100
    u = Ⱆ(10, 100, m)
    l = Ⰵ(0.5 + 0.04 * u)
    p = Ⱂ(l)
    d = Ⰴ({"user": u, "req": p})
    print(d.head)

    # Parámetros gradiente
    α = 0.000000119
    ε = 1e-6
    x = Ⰽ((Ⱁ(m), u.reshape(-1, 1)))
    β = Ⰲ([0, 0])

    # Funciones para descender el gradiente
    # ℓ(β) = Ⱄ[y * Ⰾ(λ) - λ - Ⰾ(Ⰳ(y))]
    # Ʌ = lambda t, k, λ: m * Ⰾ(k) - m * Ⰾ(λ) + Ⱄ([(k - 1) * Ⰾ(t / λ)]) - Ⱄ((t / λ) ** 2) # ???
    # Ʌ = lambda x, y, β: Ⱄ() # A perkele >:C
    λ = lambda β: Ⰵ(Ⱌ(x @ β, -50, 50))
    # J = lambda x, y, β: Ⱄ(λ(x, β) - y @ Ⰾ(λ(x, β))) # ???
    # j = lambda x, y, β: x.T@(λ(x, β) - y) # como que no
    J = lambda β: -Ⱄ(p * Ⰾ(Ⱌ(λ(β), 1e-15, None)) - λ(β))  # + Ⰳ(p))
    j = lambda β: x.T @ (λ(β) - p)
    g = lambda β, α: β - α * j(β)

    φ = [J(β), J(β)]

    for i in range(0, 1000000, 1):
        β = g(β, α)
        φ[0], φ[1] = J(β), φ[0]

        if Ⰰ(φ[0] - φ[1]) < ε:
            print(f"Convergencia alcanzada en {i} iteraciones")
            break
        if i % 10000 == 0:
            print(f"Iteración {i}: β = {β} J(β) = {φ[0]}")

    print(f"\nResultado final:")
    print(f"β estimado = [{β[0]:.6f}, {β[1]:.6f}]")
    print(f"β verdadero = [0.500000, 0.040000]")
    print(f"Error absoluto = [{abs(β[0] - 0.5):.6f}, {abs(β[1] - 0.04):.6f}]")
    print(f"J(β) final = {φ[0]:.6f}")

    # M, N = ⱞ(ⰾ(-1, 1, 100), ⰾ(1, -1, 100))
    # ⱂ.figure(figsize=(12, 10))
    # ⱂ.contour(M, N, J(Ⰲ([M, N])), cmap="viridis", alpha=0.6, levels=20)
    # ⱂ.contourf(M, N, J(Ⰲ([M, N])), cmap="viridis", alpha=0.1, levels=20)


if __name__ == "__main__":
    main()
