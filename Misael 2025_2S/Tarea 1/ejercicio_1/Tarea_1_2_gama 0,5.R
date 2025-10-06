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
gama <- 0.5
#gama <- 0.7
#gama <- 0.9
alpha <- 0.005
v <- c(0,0)
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

#malla para curva de nivel
x1_seq <- seq(-4, -1, length.out = 500)  # Rango en x1
x2_seq <- seq(1, 6, length.out = 500)  # Rango en x2
#-----------------

# 2. Calcular f(x1, x2) para cada punto de la malla
z <- matrix(NA, nrow = length(x1_seq), ncol = length(x2_seq))
for (i in 1:length(x1_seq)) {
  for (j in 1:length(x2_seq)) {
    z[i, j] <- f(c(x1_seq[i], x2_seq[j]))
  }
}
#-----------------

# Inicializar gráfico
contour(x1_seq, x2_seq, z, 
        nlevels = 30,           # Número de curvas
        col = "black",      # Color de las curvas
        lwd = 1,
        xlim = c(-3, -2), 
        ylim = c(2, 4),
        xlab = "x1", 
        ylab = "x2",
        main = "Descenso de Gradiente sobre Himmelblau")
#-----------------

# Agregar leyenda
legend("topright", legend = c("Inicio", "Trayectoria", "Final", "inicial"), 
       col = c("blue", "red", "black","orange"), pch = 20, cex = 1)
#-----------------

#Punto inicial
points(-2.5, 2.5, pch = 20, col = "orange", cex = 2)
#-----------------

for (i in 1:max_inter){
  #Cada vez que entra suma 1 para tener la cantidad de iteracion
  contador <- contador + 1
  #-----------------
  #calculo de la velocidad
  v <- (gama * v) + f_prima(x)
  #-----------------
  #calculo del nuevo X
  x_nuevo <- x - alpha*v
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
  
  #retriccion para evitar erroress
  if(norm(f_prima(x_nuevo), "2") < tolerancia | 
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
cat("el valor de Gama es: ",gama, "\n")
cat("El punto mínimo encontrado es:", x_nuevo, "\n")
cat("El valor de la función en ese punto es:", f(x_nuevo), "\n")