f <- function(x,y){
  
  return(c((x-2)^2 + (y+3)^2 ))
  
}

gradiente_f_x <- function(x){
  
  w <- 2*(x-2)
  return(w)
}

gradiente_f_y <- function(y){
  
  z <- 2*(y+3)
  return(z)
}

gradiente_f_2_x <- function(x){
  
  w<-2
  return(w)
}

gradiente_f_2_y <- function(y){
  
  z <- 2
  return(z)
}

newton_x <- function(x){
  
  m <- x
  n <- gradiente_f_x(x) / gradiente_f_2_x(x)
  b <- m - n
  return(b)
}

newton_y <- function(y){
  
  m <- y
  n <- gradiente_f_y(y) / gradiente_f_2_y(y)
  b <- m - n
  return(b)
}

valor_inicial_x <- 10
valor_inicial_y <- 10
tol <- 1e-6
max_inter <- 4
con <- 0

for (i in 1:max_inter) {
  
  con <- con+1
  valor_nuevo_x <- newton_x(valor_inicial_x)
  valor_nuevo_y <- newton_y(valor_inicial_y)
  
  if(norm(gradiente_f_x(valor_nuevo_x), "2") < tol | 
     norm(valor_nuevo_x - valor_inicial_x, "2") < tol){
    break
  }
  
  if(norm(gradiente_f_y(valor_nuevo_y), "2") < tol | 
     norm(valor_nuevo_y - valor_inicial_y, "2") < tol){
    break
  }
  
  valor_inicial_x <- valor_nuevo_x
  valor_inicial_y <- valor_nuevo_y
}

print(con)
print(valor_nuevo_x)
print(valor_nuevo_y)