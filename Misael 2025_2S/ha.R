f_f <- function(x,y){
  
  #(1 - x)^2 + 100*(y - x^2)^2
  #(1 - 2*x +x^2) + 100*(y^2 -2*y*x^2 + x^4)
  (100*x^4) + (x^2) -(2*x) - (200*y*x^2) + (100*y^2) +(1)
}

f_1x <- function(x,y){
  
  w <- 400*x^3 + 2*x -2
  return(w)
}

f_1y <- function(x,y){
  
  z <- 200*y
  return(z)
}

f_2x <- function(x,y){
  
  w <- 2
  return(w)
}

f_2y <- function(x,y){
  
  z <- 200
}

newton_x <- function(x,y){
  
  m <- x
  n <- f_1x(x,y)/f_2x(x,y)
  b <- m - n
  return(b)
}

newton_y <- function(x,y){
  
  m <- y
  n <- f_1y(x,y)/f_2y(x,y)
  b <- m - n
  return(b)
}

max_inter <- 1000
tol <- 1e-16
x <- 10
y <- 10

for (i in 1:max_inter) {
  
  valor_nuevo_x <- newton_x(x,y)
  valor_nuevo_y <- newton_y(x,y)
  
  if(norm(f_1x(x,y), "2") < tol |
     norm(valor_nuevo_x - x, "2") < tol){
    break
  }
  
  if(norm(f_1y(x,y), "2") < tol |
     norm(valor_nuevo_y - y, "2") < tol){
    break
  }
  
  x <- valor_nuevo_x
  y <- valor_nuevo_y
}

print(valor_nuevo_x)
print(valor_nuevo_y)