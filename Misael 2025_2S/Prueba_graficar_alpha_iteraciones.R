# Función objetivo (Rosenbrock)
f <- function(x){
  (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
}

# Gradiente
f_gran <- function(x){
  dx <- -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2)
  dy <- 200*(x[2] - x[1]^2)
  return(c(dx, dy))
}

# Parámetros
alpha <- 0.002
gama <- alpha/10000
x <- c(-1, -1)

# Constantes
tolerancia <- 1e-6
max_inter <- 10000

# Variables de seguimiento
contador <- 0
distancia_total <- 0  # ← INICIALIZAR AQUÍ

# Vectores para guardar histórico
trayectoria_x1 <- numeric(max_inter)
trayectoria_x2 <- numeric(max_inter)
alphas <- numeric(max_inter)

# Loop principal
for (i in 1:max_inter){
  contador <- contador + 1
  
  # Calcular alpha_k
  alpha_k <- alpha / (1 + contador * gama)
  
  # Guardar datos
  trayectoria_x1[i] <- x[1]
  trayectoria_x2[i] <- x[2]
  alphas[i] <- alpha_k
  
  # Calcular nuevo punto
  x_nuevo <- x - alpha_k * f_gran(x)
  
  # Calcular distancia
  distancia_paso <- sqrt((x_nuevo[1] - x[1])^2 + (x_nuevo[2] - x[2])^2)
  distancia_total <- distancia_total + distancia_paso
  
  # Criterio de convergencia
  if(norm(f_gran(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  
  x <- x_nuevo
}

# Agregar punto final
trayectoria_x1[contador + 1] <- x[1]
trayectoria_x2[contador + 1] <- x[2]

# Recortar vectores
trayectoria_x1 <- trayectoria_x1[1:(contador + 1)]
trayectoria_x2 <- trayectoria_x2[1:(contador + 1)]
alphas <- alphas[1:contador]

# ========== GRÁFICOS ==========
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

# Gráfico 1: Trayectoria
plot(trayectoria_x1, trayectoria_x2, 
     type = "n",  # No graficar aún
     xlim = c(-1.2, 1.2), ylim = c(-1.2, 1.5),
     xlab = "x1", ylab = "x2",
     main = "Descenso de Gradiente - Rosenbrock")

# Líneas de trayectoria
segments(trayectoria_x1[-length(trayectoria_x1)], 
         trayectoria_x2[-length(trayectoria_x2)],
         trayectoria_x1[-1], 
         trayectoria_x2[-1], 
         col = "gray70", lwd = 2)

# Puntos
points(trayectoria_x1, trayectoria_x2, pch = 20, col = "red", cex = 1)
points(trayectoria_x1[1], trayectoria_x2[1], pch = 20, col = "blue", cex = 2)
points(trayectoria_x1[length(trayectoria_x1)], 
       trayectoria_x2[length(trayectoria_x2)], 
       pch = 20, col = "black", cex = 2)

# Punto óptimo teórico (1, 1)
points(1, 1, pch = 4, col = "green", cex = 2, lwd = 2)

# Gráfico 2: Alpha vs Iteraciones
plot(1:contador, alphas, 
     type = "l", col = "blue", lwd = 2,
     xlab = "Iteración", ylab = "Alpha",
     main = "Decaimiento de Alpha")

points(1:contador, alphas, pch = 20, col = "red", cex = 0.5)

grid()

# Resultados
cat("El número de iteraciones necesarias es:", contador, "\n")
cat("El valor de alpha inicial:", alpha, "\n")
cat("El valor de alpha final:", alphas[contador], "\n")
cat("El punto mínimo encontrado es: (", x[1], ",", x[2], ")\n")
cat("El valor de la función en ese punto es:", f(x), "\n")
cat("Distancia total recorrida:", distancia_total, "\n")