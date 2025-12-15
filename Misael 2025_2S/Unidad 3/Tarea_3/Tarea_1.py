# Datos simulados pregunta 1
#-------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as mate
import scipy.special
# semilla
np.random.seed(2025)

m = 100

# Variable independiente (usuarios)
# NUMERO DE USUARIOS   = X
usuarios = np.random.uniform(10, 100, m)
# Variable dependiente (peticiones)
lambda_real = np.exp(0.5 + 0.03 * usuarios)
# NUMERO DE PETICIONES = Y
peticiones = np.random.poisson(lambda_real)
datos_pregunta_1 = pd.DataFrame({"Usuarios": usuarios,
"Peticiones": peticiones})
print(datos_pregunta_1.head())

#parametros inicales
beta = np.array([0.0, 0.0])
alpha = 0.000000333  # Tasa de aprendizaje
epsilon = 1e-6  # Tolerancia para convergencia
iteraciones = 100000
diezPor = iteraciones*0.1
tolerancia = 1e-9


# Calculo de Lambda
def Landa(usuarios, beta):
    lamda = np.exp(beta[0] + beta[1] * usuarios)
    return lamda

# Calculo de GLM
def GLM(peticiones, usuarios):

    y = np.array(peticiones)
    x = np.array(usuarios)
    Lambda= Landa(x, beta)
    y_factorial= scipy.special.factorial(y)
    P = ((np.exp(-Lambda)*Lambda**y)/y_factorial)
    return P

# Calculo del costo
def Costo_J(peticiones, usuarios, beta, epsilon):
    
    y = np.array(peticiones)
    x = np.array(usuarios)
    Lambda= Landa(x, beta)
    costo = np.sum(Lambda - y * np.log(Lambda + epsilon))
    return costo

# Calculo de la derevida 
def grand(beta, usuarios, peticiones):

    y = np.array(peticiones)
    x = np.array(usuarios)
    Lambda= Landa(x, beta)
    error =Lambda - y
    grad_beta0 = np.sum(error)
    grad_beta1 = np.sum(x * error)
    
    return grad_beta0, grad_beta1

# Descenso del gradiente
def desonse_grand(usuarios, peticiones, beta, alpha, iteraciones, tolerancia, epsilon):

    y = np.array(peticiones)
    x = np.array(usuarios)
    historial_costo = []

    for i in range(iteraciones):
        # Guardar valores actuales
        costo_actual = Costo_J(y, x, beta, epsilon)
        historial_costo.append(costo_actual)

        # Calcular gradiente
        grad_beta0, grad_beta1 = grand(beta, x, y)
        
        # Actualizar parámetros
        beta0_nuevo = beta[0] - alpha * grad_beta0
        beta1_nuevo = beta[1] - alpha * grad_beta1
        
        beta_nuevo = np.array([beta0_nuevo, beta1_nuevo])

        # Imprimir progreso cada 1000 iteraciones
        if i % (iteraciones*0.1) == 0:
            print(f"Iteración {i:5d} | Costo: {costo_actual:10.4f} | "
                  f"β₀: {beta[0]:8.5f} | β₁: {beta[1]:8.5f}")
        
        # Verificar convergencia
        cambio_beta0 = abs(beta_nuevo[0] - beta[0])
        cambio_beta1 = abs(beta_nuevo[1] - beta[1])
        
        if cambio_beta0 < tolerancia and cambio_beta1 < tolerancia:
            print(f"\n¡Convergencia alcanzada en iteración {i}!")
            break
        
        beta[0] = beta_nuevo[0]
        beta[1] = beta_nuevo[1]
    
    return beta, historial_costo

b= []
print("El valor de lambda es: ",Landa(usuarios, beta))
print("/n")
print("El valor de GLM es: ", GLM(peticiones, usuarios))
print("/n")
print("El valor del gradiente es: ",grand(beta, usuarios, peticiones))
print("/n")
print("El valor del costo es: ",Costo_J(peticiones, usuarios, beta, epsilon))
print("/n")
b,C = desonse_grand(usuarios, peticiones, beta, alpha, iteraciones, tolerancia, epsilon)
print("El valor de beta es: ",b)

usuarios_nuevos = np.array([25, 50, 75, 100])
for u in usuarios_nuevos:
    lambda_pred = Landa(u, b)
    print(f"Usuarios: {u:3d} → Tasa predicha (λ): {lambda_pred:8.2f} peticiones/min")

fig, ax = plt.subplots()
ax.plot(range(len(C)), C, label='Costo J segun Beta')
ax.set_xlabel('Iteraciones')
ax.set_ylabel('Costo J segun Beta')
ax.set_title('Evolución del Costo durante el Descenso del Gradiente')
ax.legend()
plt.show()