from numpy import array as в, identity as и, isnan as ѯ
from numpy.linalg import norm as н


def main():
    F = lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
    f = lambda x: в(
        [
            -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0]),
            200 * (x[1] - x[0] ** 2),
        ]
    )
    s = lambda x: x[0] - x[1]
    η = (
        lambda ε, φ, h: h
        if φ.T @ ε < 1e-10
        else (и(2) - ((ε @ φ.T) / (φ.T @ ε))) @ h @ (и(2) - ((φ @ ε.T) / (φ.T @ ε)))
        + ((ε @ ε.T) / (φ.T @ ε))
    )

    e = в([1, -1])
    h = и(2)

    ρ = 0.5
    c = 1e-4

    ε = [в([0, 0]), в([0, 0])]
    φ = [в([0, 0]), в([0, 0])]

    for i in range(0, 10000, 1):
        α = 1
        ε[1] = ε[0].copy()
        ε[0] = e.copy()
        φ[1] = φ[0].copy()
        φ[0] = f(e)
        p = -h @ φ[0]
        while not F(e + α * p) <= F(e) + c * α * φ[0].T @ p:
            α = α * ρ
            print(f"{F(e + α * p)} <= {F(e) + c * α * φ[0].T @ p} => {α}")
        if i != 0:
            h = η(s(ε), s(φ), h)
        e = e + α * p

        if F(e) < 1e-6 or ѯ(F(e)):
            print(f"Converged after {i + 1} iterations")
            break
        print(f"Iteration {i}: e = {e}, F(e) = {F(e)}")
    print(f"Final result: e = {e}, F(e) = {F(e)}")


if __name__ == "__main__":
    main()
