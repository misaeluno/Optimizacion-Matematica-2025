import numpy as np
import random

# f(x,y) = (x-2)**2 + (y+3)**2

def f(x):
    return (x[0] - 2)**2 + (x[1] + 3)**2

def gradiente_f(x):
    return np.array([2 * (x[0] - 2), 2 * (x[1] + 3)])

# --- PASO 1: Búsqueda de dirección (Pk) ---
def Pk(Hk, gradiente):
    # Se usa la multiplicación matricial @
    pk = -Hk @ gradiente
    return pk

# --- PASO 2: Búsqueda de línea (Armijo) ---
def alfa(xk, pk, gradiente_xk, f_xk):
    c=0.0001
    iteracion_alpha = 10
    alpha = 1  # Comienza con un tamaño de paso de 1
    
    # Condición de Armijo: f(xk + alpha*pk) <= f(xk) + c * alpha * gradiente_xk.T @ pk
    # Nota: gradiente_xk.T @ pk es el producto punto (dot product)
    grad_dot_pk = np.dot(gradiente_xk, pk)
    
    for _ in range(iteracion_alpha):
        n = xk + (alpha * pk)
        # Comparamos valores de función (escalares)
        if f(n) <= f_xk + (c * alpha * grad_dot_pk):
            break  # Condición cumplida
        alpha *= 0.5  # Reducir el paso
        
    return alpha

# --- PASO 3: Actualización de la cuasi-Hessiana (BFGS) ---
def Hk1(Hk, sk, yk):
    # s_k = x_{k+1} - x_k
    # y_k = gradiente(x_{k+1}) - gradiente(x_k)
    
    sk = sk.reshape(-1, 1) # Asegurar que es un vector columna (2x1)
    yk = yk.reshape(-1, 1) # Asegurar que es un vector columna (2x1)

    sk_T = sk.T
    yk_T = yk.T
    
    # Variables de ayuda para simplificar
    sk_yk_T = sk @ yk_T
    yk_sk_T = yk @ sk_T
    yk_T_sk = yk_T @ sk  # Producto punto, resultado escalar (1x1)

    # El denominador (rho)
    rho = 1.0 / yk_T_sk[0, 0] # Necesitamos el valor escalar

    # Término 1: Hk - (Hk @ yk @ sk.T @ Hk) / (yk.T @ Hk @ yk) + (sk @ sk.T) / (yk.T @ sk)
    
    # Término A: (I - rho * sk * yk.T) @ Hk
    I = np.identity(len(Hk))
    A = I - rho * yk_sk_T # Corrección del factor I - rho * yk * sk.T
    
    Hk1 = (I - rho * sk @ yk.T) @ Hk @ (I - rho * yk @ sk.T) + rho * sk @ sk.T
    
    # NOTA: La fórmula anterior es la versión de la inversa Hessian (H) y es la más común en la implementación.
    # Hk1 = Hk + (yk_sk_T) / (yk_T_sk) + (Hk @ yk @ yk_T @ Hk) / (yk_T @ Hk @ yk)

    return Hk1

# ================= MAIN =================
Hk = np.identity(2)
x_k = np.array([-10.0, 10.0]) # Usar float para evitar problemas
max_iteraciones = 5

for i in range(max_iteraciones):
    
    # 1. Calcular valores actuales
    f_xk = f(x_k)
    gradiente_xk = gradiente_f(x_k)
    
    # Criterio de parada: Si el gradiente es cercano a cero
    if np.linalg.norm(gradiente_xk) < 1e-6:
        print(f"\nConvergencia lograda en la iteración {i}")
        break

    # 2. Dirección de búsqueda
    pk = Pk(Hk, gradiente_xk)
    
    # 3. Búsqueda de línea (alpha)
    alpha = alfa(x_k, pk, gradiente_xk, f_xk)
    
    # 4. Nuevo punto
    x_k1 = x_k + alpha * pk
    
    # 5. Calcular sk y yk
    sk = x_k1 - x_k
    gradiente_xk1 = gradiente_f(x_k1)
    yk = gradiente_xk1 - gradiente_xk
    
    # 6. Actualizar Hk (BFGS)
    Hk = Hk1(Hk, sk, yk)
    
    # 7. Preparar para la siguiente iteración
    x_k = x_k1

    print(f"Iteración {i+1}: x = {x_k}, f(x) = {f(x_k):.6f}")

print("\n--- Resultado Final ---")
print(f"Mínimo encontrado en x: {x_k}")
print(f"Valor mínimo de f(x): {f(x_k)}")
# El resultado debe ser muy cercano al mínimo real (2, -3)