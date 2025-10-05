#funcion objetico
f <- function(x) {
  
  (x[1]^2 + x[2] -11)^2 + (x[1] + x[2]^2 -7)^2 
}
#--------------------------
#gradeinte
f_prima <- function(x){
  
  dx <- 2*(x[1]^2 + x[2] -11)*(2*x[1]) + 2*(x[1] + x[2]^2 -7)
  dy <- 2*(x[1]^2 + x[2] -11) + 2*(x[1] + x[2]^2 -7)*(2*x[2])
  return(c(dx ,dy))
}
#--------------------------
#aprendisaje
alpha <- 0.01
#alpha <- 0.001
#alpha <- 0.005
x <- c(-2.5,2.5)
#-----------------
#constantes
tolerancia <- 1e-6
max_inter <- 10000
#-----------------
#contador
contador <- 0
#-----------------
# Variable para acumular distancia total
distancia_total <- 0
#-----------------
# Inicializar gráfico
plot(x[1], x[2], xlim = c(-3, -2), ylim = c(0, 4), 
     xlab = "x", ylab = "y", main = "Descenso de Gradiente - Función Himmelblau",
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
  x_nuevo <- x - alpha*f_prima(x)
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
  if(norm(f_prima(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  #-----------------
  #remplazaomos X original por X nuevo para repetir el ciclo
  x <- x_nuevo
}

# Mostrar resultados
points(x_nuevo[1], x_nuevo[2], pch = 20, col = "black", cex = 2)
cat("El número de iteraciones necesarias es:", contador, "\n")
cat("El valor de alpha es:", alpha, "\n")
cat("El punto mínimo encontrado es:", x_nuevo, "\n")
cat("El valor de la función en ese punto es:", f(x_nuevo), "\n")