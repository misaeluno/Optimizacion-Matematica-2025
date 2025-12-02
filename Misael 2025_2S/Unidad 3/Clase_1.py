import numpy as np


def f(x) :
    
    return ((x+4)**2 + 2)

def medio(x_bueno,x_peor):

    return((x_bueno + x_peor)/2)

def reflexion(x_medio, x_bueno, alpha = 1):

    return (x_bueno + alpha*(x_bueno - x_medio))

def expansion(x_medio, xr, gama = 2):

    return (x_medio + gama*(xr - x_medio))

def encogemiento(xr):
    
    return (xr)


paso = 1
tol = 1e-6
x1 = 1
f1 = f(x1)

x_bueno=x1
f_bueno=f1

x2 = x1 + paso
f2 = f(x2)

x_peor = x2
f_peor = f2

if(f1<f2):
    x_medio=medio(x_peor,x_bueno)
    print("centroido ", x_medio)

    if(f(x_medio)<f(x_peor)):

        xr=reflexion(x_medio,x_bueno)

        print("refreccion ",xr)
        
        if(f(xr)<f(x_bueno)):

            xe=expansion(x_medio,xr)
            x_bueno = xe
            f_bueno =f(xe)
            print("expancion ",xe)
        else :

            x_bueno = xr
            f_bueno = f(xr)
    

print("x_bueno es : ", x_bueno)