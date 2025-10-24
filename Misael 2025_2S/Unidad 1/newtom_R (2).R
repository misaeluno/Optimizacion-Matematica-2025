f <- function(x){ 
  return(x^2 + 3*x -10)
}

f_prima <- function(x){
 return(2*x + 3)
}

f_prima_2 <- function(x){
  return(2)
}

tol <- 1e-16
max_inter <- 10
punto_inicial <- 10
con <-0
newton <- function(x){
  
  y <- x
  z <- f_prima(x) / f_prima_2(x)
  w <- y - z
  return(w)
}



for (i in 1:max_inter) {
  con <- con +1
  punto_nuevo <- newton(punto_inicial)
  
  if(norm(f_prima(punto_nuevo), "2") < tol | norm(punto_nuevo - punto_inicial, "2") < tol){
    break
  }

  
  punto_inicial <- punto_nuevo

}

print(con)
print(punto_nuevo)