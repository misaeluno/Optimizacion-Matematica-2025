# Base de Datos

#Frecuencia de Reloj
x[1] <- matrix(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5,
              6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
              11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
              16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
              21.5, 22.0, 22.5, 23.0, 23.5)

#Consumo de energia
x[2] <- matrix(58.319, 60.268, 66.515, 62.859, 63.780, 69.223, 66.096, 61.508,
              63.795, 65.041, 70.547, 68.434, 69.026, 68.621, 67.089, 74.595,
              71.222, 64.345, 72.895, 69.956, 68.797, 72.022, 70.331, 72.019,
              73.197, 70.956, 79.556, 78.621, 75.962, 84.457, 83.404, 82.787,
              88.029, 89.781, 91.552, 93.238, 95.070, 95.616, 97.443, 99.950,
              101.916, 106.470, 106.583, 120.370, 121.176, 118.084, 124.364,
              128.518)
#---------------------------------

#malla para curva de nivel
x1_seq <- seq(-10, 10, length.out = 500)  # Rango en x1
x2_seq <- seq(-10, 10, length.out = 500)  # Rango en x2

# 2. Calcular f(x1, x2) para cada punto de la malla
z <- matrix(NA, nrow = length(x1_seq), ncol = length(x2_seq))
for (i in 1:length(x1_seq)) {
  for (j in 1:length(x2_seq)) {
    z[i, j] <- f(c(x1_seq[i], x2_seq[j]))
  }
}

#-----------------

# Inicializar gráfico
contour(x1_seq, x2_seq, z, 
        nlevels = 30,           # Número de curvas
        col = "black",      # Color de las curvas
        lwd = 1,
        xlim = c(-3, -2), 
        ylim = c(2, 4),
        xlab = "x1", 
        ylab = "x2",
        main = "mini cudrado")
#-----------------

#valores iniciales
iteraciones <- 10000
tolerncia <- 1e-6

for (i in 1:iteraciones){
  
  
}