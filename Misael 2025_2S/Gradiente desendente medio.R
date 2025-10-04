# Función objetivo
# f(x, y) = (1 - x)^2 + 100(y - x^2)^2
rosenbrook <- function(x, y){
  return((1 - x)^2 + 100 * (y - x^2)^2)
}

x <- y <- seq(from = -1.5, to = 1.5, by = 1)
z <- outer(x, y, rosenbrook)
library(plotly)
plot_ly(x = x, y = y, z = z, type = "surface")
contour(x, y, z, nlevels = 30)

# Método descenso del gradiente
# x = c(x[1], x[2]) = (x, y)

f <- function(x){
  return((1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2)
}


# (df / dx, df / dy)
# df/dx = 2(1 - x) * -1 + 200(y - x^2) * -2x = -2(1 - x) - 400x(y - x^2)
# df/dy = 200(y - x^2)

grad_f <- function(x) {
  return(c (-2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2), 
           200*(x[2] - x[1]^2)))
}

# Parámetros del algoritmo
alpha <- 0.002
max_iter <- 500009
tol <- 0
gama <- 0.5
V<- c(0 , 0)

# Punto inicial
x_actual <- c(-1 , -1)
points(x_actual[1], x_actual[2], pch = 20, col = 2, cex = 2)

# Algoritmo
for (i in 1:max_iter){
  
  
  V<- ( (alpha * grad_f(x_actual)) + (gama*V) ) 

  x_nuevo <-( x_actual - V)
  
  points(x_actual[1], x_actual[2], pch = 20, col = "red", cex = 2)
  
  if(norm(grad_f(x_nuevo), "2") < tol | norm(x_nuevo - x_actual, "2") < tol){
    break
  }
  x_actual <- x_nuevo
}

# Mostrar resultados
resultados <- list(minimo = f(x_nuevo),
                   objetivo = x_nuevo,
                   iteraciones = i)
print(resultados)