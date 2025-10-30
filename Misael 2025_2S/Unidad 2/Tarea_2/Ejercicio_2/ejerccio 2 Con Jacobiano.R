modelo <- function(A, landa, w, teta, t) {
  
  A * exp(-landa * t) * cos(w * t + teta)
}
#-------------------------------------------------------------------------------

funcion_Objetivo <- function(beta, t_data, y_data) {
  # parametros
  A <- beta[1]
  landa <- beta[2]
  w <- beta[3]
  teta <- beta[4]
  #-----------------------------------------------------------------------------
  
  # Nuevo Y o prediccion de Y
  y_pred <- A * exp(-landa * t_data) * cos(w * t_data + teta)
  #----------------------------------------------------------------------------
  
  #
  sum((y_data - y_pred)^2)
}
#-------------------------------------------------------------------------------
calcular_jacobiana <- function(beta, t_data) {
  # Parametros
  A <- beta[1]
  landa <- beta[2]
  w <- beta[3]
  teta <- beta[4]
  #-----------------------------------------------------------------------------
  
  # Cuantos datos hay en t_data ( un "Len" de python)
  n <- length(t_data)
  #-----------------------------------------------------------------------------
  
  # Creacion de la matriz jacobiana 
  J <- matrix(0, nrow = n, ncol = 4)
  
  for(i in 1:n) {
    # rescribiendo t e Y
    t <- t_data[i]
    
    #---------------------------------------------------------------------------
    # Predicción
    exp_term <- exp(-landa * t)
    cos_term <- cos(w * t + teta)
    sin_term <- sin(w * t + teta)
    
    #---------------------------------------------------------------------------
    # Derivadas parciales respecto a cada parámetro
    J[i, 1] <- -exp_term * cos_term              #deribada de J segun A
    J[i, 2] <- A * t * exp_term * cos_term       #deribada de J segun landa
    J[i, 3] <- A * exp_term * sin_term * t       #deribada de J segun W
    J[i, 4] <- A * exp_term * sin_term           #deribada de J segun Teta
    
  }  
  return(J)
}
#-------------------------------------------------------------------------------

calcular_r <- function(beta, t_data, y_data) {
  A <- beta[1]
  landa <- beta[2]
  w <- beta[3]
  teta <- beta[4]
  #-----------------------------------------------------------------------------
  
  # Predicciones del modelo
  y_pred <- A * exp(-landa * t_data) * cos(w * t_data + teta)
  #-----------------------------------------------------------------------------
  
  # Vector de residuos
  r <- y_data - y_pred
  
  return(r)
}
#-------------------------------------------------------------------------------

gradiente <- function(beta, t_data, y_data) {
  # Calcular Jacobiana y residuos
  J <- calcular_jacobiana(beta, t_data)
  r <- calcular_r(beta, t_data, y_data)
  #-----------------------------------------------------------------------------
  
  # Gradiente: ∇S(β) = -2·J^T·r
  grad <- 2 * t(J) %*% r
  #-----------------------------------------------------------------------------
  
  # Convertir matriz (4×1) a vector
  return(as.vector(grad))
}
#-------------------------------------------------------------------------------

hessiana <- function(beta, t_data, y_data) {
  # Calcular Jacobiana
  J <- calcular_jacobiana(beta, t_data)
  #-----------------------------------------------------------------------------
  
  # Hessiana aproximada: H ≈ 2·J^T·J
  H <- 2 * t(J) %*% J
  
  return(H)
}
#-------------------------------------------------------------------------------

# Método de Newton
newton <- function(beta_0, t_data, y_data, tol = 1e-6, max_iter = 100) {
  beta <- beta_0
  historial <- matrix(NA, nrow = max_iter + 1, ncol = 5)
  colnames(historial) <- c("iter", "A", "landa", "w", "teta")
  #-----------------------------------------------------------------------------
  
  historial[1, ] <- c(0, beta)
  
  for(k in 1:max_iter) {
    grad <- gradiente(beta, t_data, y_data)
    H <- hessiana(beta, t_data, y_data)
    #---------------------------------------------------------------------------
    
    # Resolver H * delta = -grad
    delta <- tryCatch({
      solve(H, -grad)
    }, error = function(e) {
      # Si la Hessiana es singular (signo incorrecto), usar pseudo-inversa
      MASS::ginv(H) %*% (-grad)      #ginv genera la seudo-inversa
    })
    #---------------------------------------------------------------------------
    
    beta_nuevo <- beta + delta
    historial[k + 1, ] <- c(k, beta_nuevo)
    #---------------------------------------------------------------------------
    
    # Criterio de convergencia
    if(norm(delta, type = "2") < tol) {
      cat("Convergencia alcanzada en", k, "iteraciones\n")
      return(list(
        beta = beta_nuevo,
        valor_f = funcion_Objetivo(beta_nuevo, t_data, y_data),
        iteraciones = k,
        historial = historial[1:(k+1), ]
      ))
    }
    
    beta <- beta_nuevo
  }
  #-----------------------------------------------------------------------------
  
  cat("Máximo de iteraciones alcanzado\n")
  return(list(
    beta = beta,
    valor_f = funcion_Objetivo(beta_nuevo, t_data, y_data),
    iteraciones = max_iter,
    historial = historial
  ))
}

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#INT MAIN () 
# datos iniciales
set.seed(123)
A_real <- 5
landa_real <- 0.2
w_real <- 2*pi
teta_real <- pi/4
#---------------------------------------------------------------------

# Base de datos
t_data <- c(0.01585558, 0.07313676, 0.20858986, 0.33697873, 0.52537301, 0.6164369,
            0.67744082, 0.77372499, 1.24299318, 1.28798218, 1.38339642, 1.38848687,
            1.42076869, 1.46404696, 1.50057502, 1.62121603, 1.70114607, 1.94117773,
            1.96511879, 2.22784082, 2.27602643, 2.34135203, 2.34449634, 2.41064538,
            2.46308469, 2.49806005, 2.56570644, 2.59826844, 2.61573859, 3.05457166,
            3.28683793, 3.33196052, 3.80886595, 3.84728936, 4.00492237, 4.00529043,
            4.01024845, 4.23964951, 4.38247821, 4.43925851, 4.48290057, 4.56513696,
            4.58015693, 4.61305888, 4.64543839, 4.6630282, 4.73071767, 4.82119210,
            4.93766455, 4.95449219)

y_data <- c(3.30604951, 1.63304262, -2.46669298, -4.67216227, 
            -2.65183759, -0.11931208, 1.58435298, 3.20604040, -2.25300509, 
            -3.43019723, -4.02265257, -3.46359790, -3.36545235, -2.97468372, 
            -2.74451778, 0.12791535, 1.61615275, 2.82364845, 2.71400979, 
            -2.18062097, -2.39741066, -2.82599835, -3.27902725, -3.04151353, 
            -2.52012235, -2.40448523, -0.90204355, -0.69802758, -0.45049201, 
            0.99303212, -2.27801465, -2.64637196, 2.37675322, 2.28167791, 
            1.25167305, 1.60197788, 1.19320954, -1.10650603, -2.31930834, 
            -1.98134768, -1.55041113, -0.89272659, -0.75329989, -0.09842116, 
            -0.05934957, 0.67373312, 0.93814889, 1.54737425, 1.69100012,
            1.18712468)
#-------------------------------------------------------------------------------

#-----------------------------------------------------------------------
#PRUEBA
#y_data <- modelo(5, 0.2, 2*pi, pi/4, t_data) + rnorm(50, 0, 0.2)
#-----------------------------------------------------------------------
# Valores iniciales
beta0 <- c(4.5, 0.3, 6.0, 0.5)
# Ejecutar método de Newton
resultado <- newton(beta0, t_data, y_data)
#-----------------------------------------------------------------------
# VISUALIZACIÓN DE RESULTADOS
par(mfrow = c(1, 2))

# 1. Grafico 1
plot(t_data, y_data, pch = 20, col = "blue", cex = 1.2,
     xlab = "Dato t", ylab = "Dato Y", 
     main = "Grafico de dispercion datos reales")
t_plot <- seq(min(t_data), max(t_data), length.out = 200)

y_ajustado <- modelo(A_real, landa_real, 
                     w_real, teta_real, t_plot)

lines(t_plot, y_ajustado, col = "red", lwd = 2)

#legend("topright", legend = c("Datos experimentales", "Modelo ajustado"), 
#       col = c("blue", "red"), pch = c(20, NA), lty = c(NA, 1), lwd = c(1, 2))
grid()

#--------------------
#-----------------------------------------------------------------------

# 1. Grafico 1
plot(t_data, y_data, pch = 20, col = "blue", cex = 1.2,
     xlab = "Dato t", ylab = "Dato Y", 
     main = "Grafico de dispercion datos optenidos")
t_plot <- seq(min(t_data), max(t_data), length.out = 200)

y_ajustado <- modelo(resultado$beta[1], resultado$beta[2], 
                     resultado$beta[3], resultado$beta[4], t_plot)

lines(t_plot, y_ajustado, col = "red", lwd = 2)

grid()

#--------------------

precicion_a <-abs((A_real - abs(resultado$beta[1])))
# error dividio en real
precicion_a <- mean(precicion_a / A_real)

precicion_a <- precicion_a * 100
# 100% menos porcentaje ERROR
precicion_a <- 100 - precicion_a
#cat(precicion_a)


precicion_landa <-abs((landa_real - abs(resultado$beta[2])))
# error dividio en real
precicion_landa <- mean(precicion_landa / landa_real)

precicion_landa <- precicion_landa * 100
# 100% menos porcentaje ERROR
precicion_landa <- 100 - precicion_landa
#cat(precicion_landa)


precicion_w <-abs((w_real - abs(resultado$beta[3])))
# error dividio en real
precicion_w <- mean(precicion_w / w_real)

precicion_w <- precicion_w * 100
# 100% menos porcentaje ERROR
precicion_w <- 100 - precicion_w
#cat(precicion_w)


precicion_teta <-abs((teta_real - abs(resultado$beta[4])))
# error dividio en real
precicion_teta <- mean(precicion_teta / teta_real)

precicion_teta <- precicion_teta * 100
# 100% menos porcentaje ERROR
precicion_teta <- 100 - precicion_teta
#cat(precicion_teta)

precicion_total <- mean(precicion_a,precicion_landa,precicion_teta,precicion_w)
#cat(precicion_total)
#--------------------
#-----------------------------------------------------------------------


cat("\n=== RESULTADOS Con JACOBIANO===\n")
cat("A obtenido     =",resultado$beta[1],"   y A buscado    =",A_real,"\n")
cat("Su precicion es =",precicion_a,"\n")
cat("Landa obtenido =",resultado$beta[2],"   y Landa buscad =",landa_real,"\n")
cat("Su precicion es =",precicion_landa,"\n")
cat("W obtenido     =",resultado$beta[3],"  y W buscado    =",w_real,"\n")
cat("Su precicion es =",precicion_w,"\n")
cat("teta obtenido  =",resultado$beta[4],"   y teta buscado =",teta_real,"\n")
cat("Su precicion es =",precicion_teta,"\n")
cat("Iteraciones necesarias = ",resultado$iteraciones,"\n")
cat("Valor de la Funcion(β) =", resultado$valor_f, "\n")
cat("La precicion del progra es =",precicion_total)