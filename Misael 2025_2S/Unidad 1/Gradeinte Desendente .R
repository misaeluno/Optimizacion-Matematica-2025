#funcion
#     (1-x)^2 + 100(y - x^2)^2

f <- function(x,y){
  
  (1 - x)^2 + 100*(y - x^2)^2
  
}

x <- y <- seq(from=-10 , to=10, by=1)
z <- outer(x, y, f)
plot_ly(x=x, y=y, z=z, type="surface")

#---------------------------------------
# nose


#---------------------------------------
#gradiente

grad_f <- function(beta, x, y){
  
  e <- y -beta[1] - (beta[2]*x)
  d1 <- 2*(1-x)*(-1) + ((100*2)*(y-x^2)*(-2x))
  d2 <- ((100*2)*(y-x^2))
  return(c(d1,d2))
}

#-----------------------------------
#parametros iniical

alpha = 0.001
max_inter= 1000
tol <= 1e-6

#----------------------------------
#punto inicial

beta_actual <- c(0,1)

#-------------------------------------
#metodo gradiente

for ( i in 1:max_inter){
  
  beta_nuevo <- beta_actual -alpha * grad_f(beta_actual, x, y)
  
  if(norm(grad_f(beta_nuevo, x, y),"2") < tol ){
    break
  }
  
  beta_actual <- beta_nuevo
}

albine(a=beta_nueva[1],b=beta_nuevo[2],
       lwd=3, col="RED")

