# Definimos la función objetivo
f <- function(x) {
  return(0.5*(x[1]**2)  + 2.5*(x[2]**2) -2*x[1]*x[2]  - x[1]**3 )
}

#-------------------------------------------------------------------------------
# Definimos el vector gradiente de la función
gradiente_f <- function(x) {
  dx <- x[1] - 2*x[2] - 3*x[1]**2
  dy <- 5*x[2] - 2*x[1]
  return(c(dx, dy))
} 

#-------------------------------------------------------------------------------
# Definimos la matriz Hessiana de la función
hessiana_f <- function(x) {
  dxx <- 1 -6*x[1]
  dxy <- -2
  dyx <- -2
  dyy <- 5
  H <- matrix(c(dxx, dxy,
                dyx, dyy ), nrow = 2, byrow = TRUE)
  return(H)
}

#-------------------------------------------------------------------------------
# Definimos la matriz Hessiana regularizada de la función
hessiana_R <- function(x, delta = 1) {
  #llamamos a la hesiana normal
  H <- hessiana_f(x)
  #conseguimos los numeros de !!!VALORES PROPIOS!!!
  descomposicion <- eigen(H)
  
  # Ahora 'delta' está definido dentro del alcance de la función
  #C = Buscame el valor maximo de(DESCOMPOSICION y DEALTA)   
  #Evitamos que sea exponencial
  c <- pmax(descomposicion$values, delta) 
  
  I <- diag(nrow(H))
  
  # Reconstruir la matriz regularizada
  H_reg <- H - c * I
  return(H_reg)
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Parámetros del algoritmo
valor_inicial <- c(1.5, 0.5)
tol <- 1e-6
max_inter <- 10000  # Aumentamos las iteraciones por si acaso, aunque no sean necesarias aquí
iteraciones <- 0
iteraciones_r <- 0

#-------------------------------------------------------------------------------
# Bucle del método de Newton
x_k_h <- valor_inicial
x_k_r <- valor_inicial

# necesario para graficar Hessiana normal
historial_x_H <- numeric(0)
historial_y_H <- numeric(0)
historial_f_H <- numeric(0)

# necesario para graficar Hessiana regularizada
historial_x_R <- numeric(0)
historial_y_R <- numeric(0)
historial_f_R <- numeric(0)

#-------------------------------------------------------------------------------
#Hessiana regularizada
for (i in 1:max_inter) {
  
  #guardamos los parametros "nuevos" de Hessina regularizada
  historial_x_R <- c(historial_x_R, x_k_r[1])
  historial_y_R <- c(historial_y_R, x_k_r[2])
  historial_f_R <- c(historial_f_R, f(x_k_r))
  
  iteraciones_r <- iteraciones_r + 1
  
  #-----------------------------------------------------------------------------
  # Calculamos el siguiente punto
  paso_newton_R <- solve(hessiana_R(x_k_r)) %*% gradiente_f(x_k_r) #H^-1 * F
  x_k1_r <- x_k_r - paso_newton_R  
  
  # Verificar si hay valores inválidos
  if (any(is.na(x_k1_r)) || any(is.infinite(x_k1_r))) {
    cat("Error: Valores inválidos en iteración", iteraciones_r, "\n")
    cat("x_k1_r =", x_k1_r, "\n")
    break
  }
  #print(paste(x_k1_r))
  
  #-----------------------------------------------------------------------------
  # Criterios de parada
  if (all(abs(gradiente_f(x_k1_r)) < tol) || all(abs(x_k1_r - x_k_r) < tol)) {
    cat("Hessiana regularizada paró en iteración:", iteraciones_r, "\n")
    
    # Guardar punto final
    historial_x_R <- c(historial_x_R, x_k1_r[1])
    historial_y_R <- c(historial_y_R, x_k1_r[2])
    historial_f_R <- c(historial_f_R, f(x_k1_r))
    x_k_r <- x_k1_r
    break
  }
  
  #-----------------------------------------------------------------------------
  x_k_r <- x_k1_r
}

#Hessiana normal
for (i in 1:max_inter) {
  
  #guardamos los parametros "nuevos" de Hessina normal
  historial_x_H[i] <- x_k_h[1]
  historial_y_H[i] <- x_k_h[2]
  historial_f_H[i] <- f(x_k_h)
  
  iteraciones <- iteraciones + 1
  
  #-----------------------------------------------------------------------------
  # Calculamos el siguiente punto
  paso_newton_H <- solve(hessiana_f(x_k_h)) %*% gradiente_f(x_k_h) #H^-1 * F
  x_k1_h <- x_k_h - paso_newton_H                                  #X - (H^-1 * F)
  #-----------------------------------------------------------------------------
  
  # Criterios de parada
  if (all(abs(gradiente_f(x_k1_h)) < tol) || all(abs(x_k1_h - x_k_h) < tol)) {
    cat("paramos en iteracion = ",iteraciones)
    break
  }
  
  #-----------------------------------------------------------------------------
  x_k_h <- x_k1_h
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Si no esta no funciona 
#------NO BORRAR---------
historial_x_H <- historial_x_H[1:i]
historial_y_H <- historial_y_H[1:i]
historial_f_H <- historial_f_H[1:i]

# Si no esta no funciona 
#------NO BORRAR---------
historial_x_R <- historial_x_R[1:i]
historial_y_R <- historial_y_R[1:i]
historial_f_R <- historial_f_R[1:i]

#-------------------------------------------------------------------------------
# 1. Curvas de nivel para Hessiana normal
x_seq_H <- seq(-2, 2, length.out = 100)
y_seq_H <- seq(-2, 2, length.out = 100)

# 1. Curvas de nivel para Hessiana Regularizada
x_seq_R <- seq(-2, 2, length.out = 100)
y_seq_R <- seq(-2, 2, length.out = 100)

#-------------------------------------------------------------------------------
#funcion para obtener Z
f_outer <- function(x, y) {
  return(0.5*x^2 + 2.5*y^2 - 2*x*y - x^3)
}

#-------------------------------------------------------------------------------
#creacion de marco de graficos
par(mfrow = c(1, 2))

#-------------------------------------------------------------------------------
#valor de Z
z_H <- outer(x_seq_H, y_seq_H, f_outer)
z_R <- outer(x_seq_R, y_seq_R, f_outer)

#-------------------------------------------------------------------------------
#Grafico de Hessina Normal
contour(x_seq_H, y_seq_H, z_H,  
        nlevels = 30, 
        col = "lightblue",
        xlab = "x", ylab = "y",
        main = "Curvas de Nivel Hessiana normal")

# Superponer trayectoria
lines(historial_x_H, historial_y_H, col = "red", lwd = 2)
points(historial_x_H, historial_y_H, pch = 20, col = "red", cex = 1)
points(historial_x_H[1], historial_y_H[1], pch = 19, col = "green", cex = 2)
points(historial_x_H[i], historial_y_H[i], pch = 19, col = "blue", cex = 2)

#-------------------------------------------------------------------------------
#Grafico de Hessina Regularizada
contour(x_seq_R, y_seq_R, z_R,  
        nlevels = 30, 
        col = "lightblue",
        xlab = "x", ylab = "y",
        main = "Curvas de Nivel Hessiana Regularizda")

# Superponer trayectoria
lines(historial_x_R, historial_y_R, col = "red", lwd = 2)
points(historial_x_R, historial_y_R, pch = 20, col = "red", cex = 1)
points(historial_x_R[1], historial_y_R[1], pch = 19, col = "green", cex = 2)
points(historial_x_R[i], historial_y_R[i], pch = 19, col = "blue", cex = 2)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

a_precicion <- (resultado$beta[1]-beta0[1]) / beta0[1]
landa_precicion <- (beta0[2]-resultado$beta[2]) / beta0[2]
w_precicion <- (resultado$beta[3]-beta0[3]) / beta0[3]
teta_precicion <- (resultado$beta[4]-beta0[4]) / beta0[4]

#cat(a_precicion,"/ /",landa_precicion,"/ /",w_precicion,"/ /",teta_precicion)

a_precicion <- a_precicion*(100)
landa_precicion <- landa_precicion*(100)
w_precicion <- w_precicion*(100)
teta_precicion <- teta_precicion*(100)

cat("Precicion de A ",a_precicion,"%")
cat("Precicion de Landa ",landa_precicion,"%")
cat("Precioin de Omega ",w_precicion,"%")
cat("Precicionj de Phi",teta_precicion,"%")

a_precicion <- a_precicion*(0.25)
landa_precicion <- landa_precicion*(0.25)
w_precicion <- w_precicion*(0.25)
teta_precicion <- teta_precicion*(0.25)

precicion_total <- 100 - (a_precicion+landa_precicion+w_precicion+teta_precicion)

cat("Porcentaje de la precicion general del programa ",precicion_total,"%")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

# Resultados
cat("\n=== RESULTADOS ===\n")
cat("\nHessiana Normal:\n")
cat("  Iteraciones:", iteraciones, "\n")
cat("  Mínimo encontrado en x:", x_k_h[1], "\n")
cat("  Mínimo encontrado en y:", x_k_h[2], "\n")
cat("  Valor de la función:", f(x_k_h), "\n")

cat("\nHessiana Regularizada:\n")
cat("  Iteraciones:", iteraciones_r, "\n")
cat("  Mínimo encontrado en x:", x_k_r[1], "\n")
cat("  Mínimo encontrado en y:", x_k_r[2], "\n")
cat("  Valor de la función:", f(x_k_r), "\n")