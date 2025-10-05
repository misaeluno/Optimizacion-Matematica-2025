#funcion objetivo
f <- function (x){
  (1 - x[1])^2 + 100*( x[2] - x[1]^2)^2
}

f_gran <- function(x){
  dx <- -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2)
  dy <- 200*(x[2] - x[1]^2)
  return(c(dx,dy))
}

#alfa
alpha <- 0.002
#valor inicial
x <- c(-1,-1)
#-----------------
#constantes
tolerancia <- 1e-6
max_inter <- 10000
#-----------------
#contador
contador <- 0

# Inicializar gráfico
plot(x[1], x[2], xlim = c(0.25, 1.5), ylim = c(0.01,2), 
     xlab = "x", ylab="y log",
     log = "y",
     main = "Descenso de Gradiente - Función de Rosenbrock",
     pch = 20, col = "blue", cex = 2)
#-----------------

# Agregar leyenda
legend("topright", legend = c("Inicio", "Trayectoria", "Final"), 
       col = c("blue", "red", "black"), pch = 20, cex = 1)
#-----------------

for (i in 1:max_inter){
  #Cada vez que entra suma 1 para tener la cantidad de iteracion
  contador <- contador + 1
  #-----------------
  #calculo del nuevo X
  x_nuevo <- x - alpha*f_gran(x)
  #-----------------
  #graficar
  points(x_nuevo[1], x_nuevo[2], pch = 20, col = "red", cex = 2)
  #-------------------------------------------------------------------
  # Graficar línea de trayectoria entre x y x_nuevo
  segments(x[1], x[2], x_nuevo[1], x_nuevo[2], col = "gray70", lwd = 4)
  #-----------------
  # Calcular distancia recorrida en este paso
  distancia_paso <- sqrt((x_nuevo[1] - x[1])^2 + (x_nuevo[2] - x[2])^2)
  distancia_total <- distancia_total + distancia_paso
  #-------------------------------------------------------------------
  #retriccion para evitar errores
  if(norm(f_gran(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  #-----------------
  #remplazaomos X original por X nuevo para repetir el ciclo
  x <- x_nuevo
}
points(x_nuevo[1], x_nuevo[2], pch = 20, col = "black", cex = 2)
cat("El número de iteraciones necesarias es:", contador, "\n")
cat("El valor de alpha es:", alpha, "\n")
cat("El punto mínimo encontrado es:", x_nuevo, "\n")
cat("El valor de la función en ese punto es:", f(x), "\n")