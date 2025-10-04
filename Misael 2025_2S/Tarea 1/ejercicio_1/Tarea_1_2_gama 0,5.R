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
  #retriccion para evitar erroress
  if(norm(f_prima(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  #-----------------
  #remplazaomos X original por X nuevo para repetir el ciclo
  x <- x_nuevo
}
cat("el nuemro de interaciones necesaria es: ",contador, "\n")
cat("el valor de Gama es: ",gama, "\n")
cat("el valor del punto es: ",x)