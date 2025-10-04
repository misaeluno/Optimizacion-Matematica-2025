# FUNCIONES CLAVE
# ---------------

# Función objetivo
f_obj <- function(x) {
  return ((x[1] - 2)**2 + (x[2] + 3)**2)
}

# Función gradiente
f_grad <- function(x) {
  return (c(2*(x[1] - 2), 2*(x[2] + 3)))
}

# FUNCIONES FUNDAMENTALES
# -----------------------

p_k <- function(hess_k, x_k)
  return( -hess_k %*% f_grad(x_k) )

x_k_mas_1 <- function(x_k, alpha_k, p_k) {
  return (x_k + alpha_k * p_k)
}

# MI IMPLEMENTACIÓN DE hess_k+1
# -----------------------------

# hess_k_mas_1 <- function(s_k, y_k, hess_k) {
#   return ((identidad - (s_k*t(y_k))/(t(y_k)*s_k)) * hess_k*(identidad - (y_k*t(s_k))/(t(y_k)*s_k)) + ((s_k)*t(s_k))/(t(y_k)*s_k))
# }

# IMPLEMENTACIÓN DE CHATGPT
# -------------------------

hess_k_mas_1 <- function(s_k, y_k, hess_k) {
  rho <- 1 / as.numeric(t(y_k) %*% s_k)
  I <- diag(length(s_k))
  term1 <- (I - rho * s_k %*% t(y_k))
  term2 <- (I - rho * y_k %*% t(s_k))
  H_new <- term1 %*% hess_k %*% term2 + rho * (s_k %*% t(s_k))
  return(H_new)
}

s_k <- function(x_k_mas_1, x_k)
  return (x_k_mas_1 - x_k)

y_k <- function(x_k_mas_1, x_k)
  return (f_grad(x_k_mas_1) - f_grad(x_k))

# OBTENCIÓN DE ALPHA
# ------------------

condicion_armijo <- function(x_k, alpha_k, p_k, c) {
  if (!(f_obj(x_k + alpha_k * p_k) <= f_obj(x_k) + c*alpha_k*t(f_grad(x_k)) %*% p_k)) {
    alpha_k <- alpha_k * factor_retroceso
  }
  return (alpha_k)
}

# FUNCIÓN DEL MÉTODO QUASI-NEWTON
# -------------------------------

quasi_newton <- function(x_actual, alpha_actual, hess_actual, grad_actual) {
  x_historial <- x_actual
  for (i in 1:max_iter) {
    if (i == 1) {
      p_actual <- p_k(identidad, x_actual)
      alpha_actual <- condicion_armijo(x_actual, alpha_actual, p_actual, c)
      x_nuevo <- x_k_mas_1(x_actual, alpha_actual, p_actual)
      x_historial <- rbind(x_historial, matrix(x_nuevo, nrow = 1))
      hess_actual <- identidad
      next
    }
    s_nuevo <- s_k(x_nuevo, x_actual)
    y_nuevo <- y_k(x_nuevo, x_actual)
    
    hess_nuevo <- hess_k_mas_1(s_nuevo, y_nuevo, hess_actual)
    p_nuevo <- p_k(hess_nuevo, x_nuevo)
    alpha_nuevo <- condicion_armijo(x_nuevo, alpha_actual, p_nuevo, c)
    x_nuevo <- x_k_mas_1(x_nuevo, alpha_nuevo, p_nuevo)
    x_historial <- rbind(x_historial, matrix(x_nuevo, nrow = 1))
    
    if ((norm(f_grad(x_nuevo), "2") < tol) || 
        (norm(x_nuevo - x_actual, "2") < tol)) {
      break
    }
    x_actual <- x_nuevo
  }
  return (x_historial)
}

# CONSTANTES
# ----------
x_0 <- c(-10, 10)
alpha_0 <- 1
identidad <- matrix(c(1, 0, 0 ,1) , nrow = 2, byrow = TRUE)
c <- 1e-4
factor_retroceso <- .5

max_iter = 10000
tol <- 1e-6

resultado <- quasi_newton(x_0, alpha_0, identidad, f_grad(x_0))



  
  
  
  
  
  
  