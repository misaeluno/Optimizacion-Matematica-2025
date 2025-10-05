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
alpha_k <- 0

#gama
gama <-alpha/10000

#valor inicial
x <- c(-1,-1)

#constantes
tolerancia <- 1e-6
max_inter <- 10000

#contador
contador <- 0
distancia_total <- 0 

#-----------------------------------------------------------------
plot(contador, alpha_k,
     xlim = c(0, max_inter), ylim = c(0.00199,0.002), 
     xlab = "Iteraciones", ylab = "Alpha",
     main = "Decaimiento de Alpha",
     type = "n")

#-----------------------------------------------------------------
for (i in 1:max_inter){
  
  #Cada vez que entra suma 1 para tener la cantidad de iteracion
  contador <- contador + 1
  #-----------------
  #calcular alfa k
  alpha_k <- (alpha)/(1 + contador*gama)
  #calculo del nuevo X
  x_nuevo <- x - alpha_k*f_gran(x)
  #-----------------
  
  #graficar

  points(contador, alpha_k, pch = 20, col = "red", cex = 1)
  
  #retriccion para evitar errores
  if(norm(f_gran(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  #-----------------
  #remplazaomos X original por X nuevo para repetir el ciclo
  x <- x_nuevo
}
points(contador, alpha_k, pch = 20, col = "black", cex = 2)
cat("El número de iteraciones necesarias es:", contador, "\n")
cat("El valor de alpha inicial:", alpha, "\n")
cat("El valor de alpha final:", alpha_k, "\n")