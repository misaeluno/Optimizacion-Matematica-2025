f <- function(x,y,z){
  (x)^3 + (y)^3 (z)^3
}


dx <- function(x){
  #x^2 + 4*x + 4
  w <- 3*(x^2) 
  return(w)
}

dy <- function(y){
  #y^2 + 6*y + 9
  w <- 3*(y^2) 
  return(w)   
}

dz <-function(z){
  #z^2 - 4*z + 2
  w <- 3*(z^2) 
  return(w)
}

dxx <- function(x){
  6*x
}
dyy <- function(y){
  6*y
}
dzz <- function(z){
  6*z
}

netonX <- function(x){
  x - ( dx(x)/dxx(x))
}

netonY <- function(y){
  y - (dy(y)/dyy(y))
}

netonZ <- function(z){
  z - (dz(z)/dzz(z))
}

x <- 10
y <- 10
z <- 10

max<- 100

for (i in 1:max) {
 
  nuevoX<-netonX(x)
  nuevoY<-netonY(y)
  nuevoZ<-netonZ(z)
  
  x <-nuevoX
  y <-nuevoY
  z <-nuevoZ
  
}


print(x)
print(y)
print(z)