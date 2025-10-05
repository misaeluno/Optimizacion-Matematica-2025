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
#alpha <- 0.01
alpha <- 0.001
#alpha <- 0.005
x <- c(-2.5,2.5)
#-----------------
#constantes
tolerancia <- 1e-6
max_inter <- 10000
#-----------------
#contador
contador <- 0

for (i in 1:max_inter){
  #Cada vez que entra suma 1 para tener la cantidad de iteracion
  contador <- contador + 1
  #-----------------
  #calculo del nuevo X
  x_nuevo <- x - alpha*f_prima(x)
  #-----------------
  #retriccion para evitar errores
  if(norm(f_prima(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  #-----------------
  #remplazaomos X original por X nuevo para repetir el ciclo
  x <- x_nuevo
}
cat("El número de iteraciones necesarias es:", contador, "\n")
cat("El valor de alpha es:", alpha, "\n")
cat("El punto mínimo encontrado es:", x_nuevo, "\n")
cat("El valor de la función en ese punto es:", f(x_nuevo), "\n")