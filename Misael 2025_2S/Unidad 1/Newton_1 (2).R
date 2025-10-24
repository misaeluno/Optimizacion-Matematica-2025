f <- function(x,y){
  x^2 + x*y +y^2
}

f_x <- function(x,y){
  return(2*x + y )
}

f_y <- function(x,y){
  return(2*y + y )
}

f_xx <- function(x){
  2
}

f_yy <- function(x){
  2
}

new_x <- function(x,y){
  w <- x
  z <- f_x(w,y)/f_xx(w)
  b <- w - z
  return(b)
}

new_y <- function(x,y){
  w <- y
  z <- f_y(x,y)/f_yy(y)
  b <- w - z
  return(b)
}

max <- 100
x <- 10
y <- 10
con <- 0
tol <- 1e-15

for (i in 1:max){
  con <- con + 1
  
  x_nuevo <- new_x(x,y)
  y_nuevo <- new_y(x,y) 
  
  if(norm(f_x(x,y), "2") < tol | 
     norm(x_nuevo - x, "2") < tol){
    break
  }
  
  if(norm(f_y(x,y), "2") < tol | 
     norm(y_nuevo - y, "2") < tol){
    break
  }
  
  x <- x_nuevo
  y <- y_nuevo
}

print(x)
print(y)
print (con)