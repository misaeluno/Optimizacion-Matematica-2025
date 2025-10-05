# Función objetivo (Rosenbrock)
f <- function(x){
  (1 - x[1])^2 + 100*(x[2] - x[1]^2)^2
}

# Gradiente
f_gran <- function(x){
  dx <- -2*(1 - x[1]) - 400*x[1]*(x[2] - x[1]^2)
  dy <- 200*(x[2] - x[1]^2)
  return(c(dx, dy))
}

# Parámetros
alpha <- 0.002

# Valor inicial
x <- c(-1, -1)

# Constantes
tolerancia <- 1e-6
max_inter <- 10000

# Contador
contador <- 0

# ✅ IMPORTANTE: Inicializar vector para guardar historial de f(x)
historial_f <- c(f(x))  # Guardar valor inicial
historial_x <- matrix(x, nrow = 1)  # Guardar trayectoria (opcional)

# Distancia total
distancia_total <- 0

cat("\n========== INICIANDO OPTIMIZACIÓN ==========\n")
cat("Función: Rosenbrock\n")
cat("Punto inicial: (", x[1], ",", x[2], ")\n")
cat("f(x0) =", f(x), "\n")
cat("Alpha =", alpha, "\n")
cat("============================================\n\n")

# ==================== BUCLE DE OPTIMIZACIÓN ====================
for (i in 1:max_inter){
  # Incrementar contador
  contador <- contador + 1
  
  # Calcular nuevo X
  x_nuevo <- x - alpha * f_gran(x)
  
  # ✅ GUARDAR el valor de f(x_nuevo) en cada iteración
  historial_f <- c(historial_f, f(x_nuevo))
  historial_x <- rbind(historial_x, x_nuevo)
  
  # Calcular distancia
  distancia_paso <- sqrt((x_nuevo[1] - x[1])^2 + (x_nuevo[2] - x[2])^2)
  distancia_total <- distancia_total + distancia_paso
  
  # Mostrar progreso cada 2000 iteraciones
  if (contador %% 2000 == 0) {
    cat("Iteración", contador, ": f(x) =", format(f(x_nuevo), scientific = TRUE), "\n")
  }
  
  # Condición de parada
  if(norm(f_gran(x_nuevo), "2") < tolerancia | 
     norm(x_nuevo - x, "2") < tolerancia){
    break
  }
  
  # Actualizar X
  x <- x_nuevo
}

cat("\n========== OPTIMIZACIÓN COMPLETADA ==========\n\n")

# ==================== RESULTADOS ====================
cat("========== RESULTADOS ==========\n")
cat("Iteraciones:", contador, "\n")
cat("Alpha:", alpha, "\n")
cat("Punto inicial: (-1, -1)\n")
cat("Punto final: (", round(x_nuevo[1], 6), ",", round(x_nuevo[2], 6), ")\n")
cat("Mínimo global: (1, 1)\n")
cat("f(x) inicial:", format(historial_f[1], scientific = FALSE), "\n")
cat("f(x) final:", format(f(x_nuevo), scientific = TRUE), "\n")
cat("Distancia total recorrida:", round(distancia_total, 4), "\n")
cat("Mejora:", round((historial_f[1] - f(x_nuevo))/historial_f[1] * 100, 2), "%\n")
cat("================================\n")

# ==================== GRÁFICOS ====================
# Configurar ventana con 2 gráficos
par(mfrow = c(1, 2), mar = c(5, 5, 4, 2))

# ======== GRÁFICO 1: EVOLUCIÓN DE f(x,y) CON ESCALA LOGARÍTMICA ========
plot(0:(length(historial_f)-1), historial_f,
     type = "l", 
     col = "blue", 
     lwd = 2.5,
     log = "y",  # ✅ ESCALA LOGARÍTMICA EN EJE Y
     xlab = "Iteraciones", 
     ylab = "f(x, y) [escala logarítmica]",
     main = "Evolución de la Función Objetivo\nDescenso de Gradiente",
     cex.lab = 1.2,
     cex.main = 1.1,
     col.main = "darkblue")

# Agregar línea de tolerancia
abline(h = tolerancia, col = "red", lty = 2, lwd = 2)
text(length(historial_f) * 0.7, tolerancia * 5, 
     paste("Tolerancia =", format(tolerancia, scientific = TRUE)), 
     col = "red", cex = 0.9)

# Agregar línea del valor óptimo
abline(h = 0, col = "darkgreen", lty = 2, lwd = 2)
text(length(historial_f) * 0.7, 1e-10, 
     "Óptimo: f(1,1) = 0", 
     col = "darkgreen", cex = 0.9)

# Marcar punto inicial y final
points(0, historial_f[1], pch = 19, col = "darkgreen", cex = 2)
points(length(historial_f)-1, historial_f[length(historial_f)], 
       pch = 19, col = "red", cex = 2)

# Agregar cuadrícula
grid(col = "gray70")

# Leyenda
legend("topright", 
       legend = c("f(x,y)", "Inicio", "Final", "Tolerancia"),
       col = c("blue", "darkgreen", "red", "red"),
       pch = c(NA, 19, 19, NA),
       lty = c(1, NA, NA, 2),
       lwd = c(2.5, NA, NA, 2),
       cex = 0.9,
       bg = "white")

# ======== GRÁFICO 2: TRAYECTORIA EN ESPACIO (x, y) ========
plot(historial_x[,1], historial_x[,2],
     type = "l",
     col = "blue",
     lwd = 2,
     xlab = "x",
     ylab = "y",
     main = "Trayectoria en el Espacio (x, y)",
     cex.lab = 1.2,
     cex.main = 1.1,
     col.main = "darkblue",
     xlim = c(-1.2, 1.2),
     ylim = c(-1.2, 1.2))

# Marcar puntos importantes
points(-1, -1, pch = 19, col = "darkgreen", cex = 2)  # Inicio
points(1, 1, pch = 4, col = "red", cex = 2.5, lwd = 3)  # Óptimo
points(x_nuevo[1], x_nuevo[2], pch = 19, col = "black", cex = 2)  # Final

# Agregar flechas para mostrar dirección
n_arrows <- min(20, nrow(historial_x)-1)
arrow_indices <- seq(1, nrow(historial_x)-1, length.out = n_arrows)
for (idx in arrow_indices) {
  i <- round(idx)
  arrows(historial_x[i,1], historial_x[i,2],
         historial_x[i+1,1], historial_x[i+1,2],
         length = 0.08, col = "blue", lwd = 1)
}

# Cuadrícula
grid(col = "gray70")

# Leyenda
legend("topleft",
       legend = c("Inicio (-1,-1)", "Trayectoria", "Final", "Óptimo (1,1)"),
       col = c("darkgreen", "blue", "black", "red"),
       pch = c(19, NA, 19, 4),
       lty = c(NA, 1, NA, NA),
       lwd = c(NA, 2, NA, 3),
       cex = 0.9,
       bg = "white")

# ==================== ANÁLISIS ESTADÍSTICO ====================
cat("\n========== ANÁLISIS ESTADÍSTICO ==========\n")

# Reducción por rangos de iteraciones
rangos <- list(
  c(1, 100),
  c(101, 1000),
  c(1001, min(5000, length(historial_f))),
  c(min(5001, length(historial_f)), length(historial_f))
)

cat("\nReducción de f(x) por rangos de iteraciones:\n")
for (rango in rangos) {
  if (rango[2] <= length(historial_f) && rango[1] < rango[2]) {
    reduccion <- historial_f[rango[1]] - historial_f[rango[2]]
    pct <- (reduccion / historial_f[rango[1]]) * 100
    cat(sprintf("  Iters %d-%d: %.2e → %.2e (reducción: %.1f%%)\n",
                rango[1]-1, rango[2]-1,
                historial_f[rango[1]], historial_f[rango[2]], pct))
  }
}

# Velocidad de convergencia promedio
velocidad_promedio <- (historial_f[1] - historial_f[length(historial_f)]) / 
  (length(historial_f) - 1)
cat("\nVelocidad de convergencia promedio:", 
    format(velocidad_promedio, scientific = TRUE), "por iteración\n")

cat("\n==========================================\n")
