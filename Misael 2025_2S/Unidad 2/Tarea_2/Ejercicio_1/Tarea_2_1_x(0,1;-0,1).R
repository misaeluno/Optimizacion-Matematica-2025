f_objetivo <- function(x,y){
  return (2*(x**2) - 1.05*(x**4) + ((x**6)/6) + x*y + y**2 )
}

# definir parametros
f <- function(param){
  x <- param[1]
  y <- param[2]
  
  return(f_objetivo(x,y))
}

# definimos  el tirnagulo F
gran_f <- function(param){
  x <- param[1]
  y <- param[2]
  
  dx <- 4*x - (1.05*4*(x**3)) + (x**5) + y
  dy <- x + 2*y
  
  return(c(dx,dy))
}

# Definimos la matriz hessianade la función
hess_f<-function(param){
  x <-param[1]
  y <-param[2]
  
  # Segundas derivadas
  d2x <- 4 - 1.05*12*(x**2) + 5*(x**4) 
  d2y <- 2
  dxdy <- 1
  
  return(matrix(c(d2x, dxdy, 
                  dxdy, d2y),
                ncol = 2, byrow = TRUE))
}
# Establecemos losparámetros delalgoritmo
max_iter<-1000
tol <-1e-6
# Establecemos unpunto inicial
x_actual<-c(0.1,-0.2)

# necesario para graficar
historial_x <- numeric(max_iter)
historial_y <- numeric(max_iter)
historial_f <- numeric(max_iter)

for (i in 1:max_iter) {
  
  #guardamos los parametros "nuevos" 
  historial_x[i] <- x_actual[1]
  historial_y[i] <- x_actual[2]
  historial_f[i] <- f(x_actual)
  
  # X - ( hessiano **-1  * gradiente f )
  x_nuevo<-x_actual - solve(hess_f(x_actual)) %*% gran_f(x_actual)
  
  if (norm(gran_f(x_nuevo), "2")< tol){
    cat("Convergió en iteración:", i, "\n")
    break
  }
  
  x_actual<-x_nuevo
}

# Si no esta no funciona 
#------NO BORRAR---------
historial_x <- historial_x[1:i]
historial_y <- historial_y[1:i]
historial_f <- historial_f[1:i]

# GRAFICAMOS 
# Crear curvas de nivel
x_seq <- seq(-0.1, 0.2, length.out = 100)
y_seq <- seq(-0.4, 0.2, length.out = 100)
z <- outer(x_seq, y_seq, f_objetivo)

contour(x_seq, y_seq, z, 
        nlevels = 30, 
        col = "lightblue",
        xlab = "x", ylab = "y",
        main = "Trayectoria sobre Curvas de Nivel")

# Superponer trayectoria
lines(historial_x, historial_y, col = "red", lwd = 2)
points(historial_x, historial_y, pch = 20, col = "red", cex = 1)
points(historial_x[1], historial_y[1], pch = 19, col = "green", cex = 2)
points(historial_x[i], historial_y[i], pch = 19, col = "blue", cex = 2)

legend("topright", 
       legend = c("0.1 , -0.2", "Trayectoria", "Final"),
       col = c("green", "red", "blue"),
       pch = c(19, 20, 19),
       cex = 0.8)


#---------------------------------------
# Mostramos los resultados
resultados<-list(punto_minimo = as.vector(x_nuevo),
                 valor_minimo_Fx = f(x_nuevo),
                 iteraciones= i,
                 gradiente = norm(gran_f(x_nuevo), "2")
)

print(resultados)

