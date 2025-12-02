import numpy as np

# 1 funcion objetivo
def f(x):
    return (x-4)**2 +2 

# algoritmo nelder mead
def nelder_mead(f, x_star, step = 1.0, alpha = 1.0 ,gama = 2.0 ,rho =0.5, max_iter = 1000, tol=1e-6):
    #necesitamos (n+1) vertices =2
    p1=x_star
    p2=x_star +step

    #h historial
    historial = [(p1,p2)]

    for _ in range(max_iter):
        # 1 ordenar
        f1, f2 = f(p1), f(p2)
        if f1 < f2 :
            xb = p1 # El mejor
            xw = p2 # El peor
            fb = f1
            fw = f2
        else:
            xb = p2 # El mejor
            xw = p1 # El peor
            fb = f2
            fw = f1
        
        # guradad estado
        historial.append((xb,xw))

        #criterio de parada
        if np.abs(fb - fw) < tol:
            break

        # centroide
        xo = xb

        #reflexion
        xr = xo + alpha*(xo - xw)
        fr = f(xr)

        if fr < fb:
            # expancion
            xe = xo + gama*(xr - xo)
            fe = f(xe)

            if fe < fr:
                xw = xe # aceptar expancion

            else:
                xw = xr # aceptar refleccion

        else : # fr >= fb
            #contraccion
            xc = xo - rho*(xo - xw)
            fc = f(xc)

            if fc < fw:
                xw=xc #acepto la contraccion
            
            else:
                xw= (xw +xb)/2

        p1, p2 = xb, xw

        historial.append((p1,p2))
    
    return historial

reultado = nelder_mead(f,x_star=1)
print(reultado)