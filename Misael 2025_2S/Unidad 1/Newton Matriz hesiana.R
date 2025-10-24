# Definimos la función objetivo
f <- function(x_vec) {
  return((x_vec[1] - 2)^2 + (x_vec[2] + 3)^2)
}

# Definimos el vector gradiente de la función
gradiente_f <- function(x_vec) {
  dx <- 2 * (x_vec[1] - 2)
  dy <- 2 * (x_vec[2] + 3)
  return(c(dx, dy))
}

# Definimos la matriz Hessiana de la función
hessiana_f <- function(x_vec) {
  H <- matrix(c(2, 0, 0, 2), nrow = 2, byrow = TRUE)
  return(H)
}

# Parámetros del algoritmo
valor_inicial <- c(10, 10)
tol <- 1e-6
max_inter <- 10  # Aumentamos las iteraciones por si acaso, aunque no sean necesarias aquí
iteraciones <- 0

# Bucle del método de Newton
x_k <- valor_inicial
for (i in 1:max_inter) {
  
  iteraciones <- iteraciones + 1
  
  # Calculamos el siguiente punto
  paso_newton <- solve(hessiana_f(x_k)) %*% gradiente_f(x_k)
  x_k1 <- x_k - paso_newton
  
  # Criterios de parada
  if (all(abs(gradiente_f(x_k1)) < tol) || all(abs(x_k1 - x_k) < tol)) {
    break
  }
  
  x_k <- x_k1
}

# Resultados
print(paste("Iteraciones:", iteraciones))
print(paste("Mínimo encontrado en x:", x_k[1]))
print(paste("Mínimo encontrado en y:", x_k[2]))
print(paste("Valor de la función en el mínimo:", f(x_k)))