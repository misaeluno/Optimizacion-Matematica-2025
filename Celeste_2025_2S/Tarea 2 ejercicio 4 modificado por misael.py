from numpy import array as в, identity as и, isnan as ѯ
from numpy.linalg import norm as н

def funcion(x):

    return в([ -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0]),200 * (x[1] - x[0] ** 2)])

def gradiente(x):
    
    return в([ -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0]),200 * (x[1] - x[0] ** 2)])

def main():

    s = lambda x: x[0] - x[1]

    η = (
        lambda ε, φ, hessiana: hessiana
        
        if φ.T @ ε < 1e-10
        
        else (и(2) - ((ε @ φ.T) / (φ.T @ ε))) @ hessiana @ (и(2) - ((φ @ ε.T) / (φ.T @ ε)))
        + ((ε @ ε.T) / (φ.T @ ε))
    )

    x = в([1, -1])
    hessiana = и(2)

    pk = 0.5
    tolerancia = 1e-4

    ε = [в([0, 0]), в([0, 0])]
    φ = [в([0, 0]), в([0, 0])]

    for i in range(0, 10000, 1):
        alpha = 1
        ε[1] = ε[0].copy()
        ε[0] = x.copy()
        φ[1] = φ[0].copy()
        φ[0] = gradiente(x)
        pk = -hessiana @ gradiente(x)

        while not funcion(x + alpha * pk) <= funcion(x) + tolerancia * alpha * φ[0].T @ pk:

            alpha = alpha * pk
            ##### print(funcion"{funcion(x + alpha * p)} <= {funcion(x) + tolerancia * alpha * φ[0].T @ p} => {alpha}")

        if i != 0:
            hessiana = η(s(ε), s(φ), hessiana)
        x = x + (alpha * pk)

        if funcion(x) < 1e-6 or ѯ(funcion(x)):
            print(f"Converged after {i + 1} iterations")
            break
        print(f"Iteration {i}: x = {x}, funcion(x) = {funcion(x)}")
    print(f"Final result: x = {x}, funcion(x) = {funcion(x)}")


if __name__ == "__main__":
    main()
