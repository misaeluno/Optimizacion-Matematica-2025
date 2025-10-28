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
hessiana_R <- function(x, c_base = 1e-4) {
  H <- hessiana_f(x)
  
  # Calcular valores propios para verificar si la matriz es definida positiva
  valores_propios <- eigen(H, only.values = TRUE)$values
  
  # Si algún valor propio es <= 0, necesitamos regularizar
  if (any(valores_propios <= 0)) {
    # Encontrar el valor propio más negativo
    min_eigenval <- min(valores_propios)
    
    # Elegir c suficientemente grande para hacer la matriz positiva definida
    # c debe ser mayor que |min_eigenval| más un pequeño margen
    c <- abs(min_eigenval) + c_base
    
    # Regularizar: H_mod = H + c*I
    I <- diag(nrow(H))
    H_modificada <- H + c * I
    
    cat(sprintf("  [Regularización aplicada: c = %.6f, λ_min = %.6f]\n", c, min_eigenval))
    
    return(H_modificada)
  } else {
    # La Hessiana ya es definida positiva, no necesita regularización
    return(H)
  }
}
#-------------------------------------------------------------------------------
# Parámetros del algoritmo
valor_inicial <- c(1.5, 0.5)# Definimos la función objetivo
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
  H <- hessiana_f(x)
  descomposicion <- eigen(H)
  
  # Ahora 'delta' está definido dentro del alcance de la función
  valores_propios_reg <- pmax(descomposicion$values, delta) 
  
  # Reconstruir la matriz regularizada
  H_reg <- H - valores_propios_reg
  return(H_reg)
}
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