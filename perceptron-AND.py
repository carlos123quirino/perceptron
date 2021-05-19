import time
import numpy as np
import matplotlib.pyplot as plt
# Declarar vector entradas de la cpmpuerta AND
x= np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
x = x.T
 
# Declarar vector solucion de la compuerta AND
t = np.array([[0], [1], [ 1], [1]])
 
# Declarar Bias
bias  = np.full((1,4), -1)
 
# Concatenar "x" y "bias"
conca = np.concatenate((x, bias), axis=0)
print("Matriz")
print (conca.T)
 
# Crear verctor de pesos
weights_0 = np.random.random_sample((3,1))
print("Pesos aleatorios")
print(weights_0)
 
# Declarar alpha
alpha = 0.05
# Epocas generar epocas
epocas = 0
a = conca.T.dot(weights_0)
inc_w = 1
def incrementoPeso(weights_0):
    a = conca.T.dot(weights_0)
    y = np.heaviside(a,0)
    # incremento de W
    global inc_w
    inc_w = (conca).dot(t-y)
    weights_1 = weights_0 + inc_w*alpha
    return (weights_1)

timer = time.time()
while (np.count_nonzero(inc_w) != 0 ):
    weights_0 = incrementoPeso(weights_0)
    epocas += 1
    print(weights_0)
timer = time.time() - timer
# Graficar
if t[0] == 0:
    mark = 'o'
else:
    mark = 'x'  
    
plt.plot(x[0,0] , x[1,0],c='r', marker = mark)

if t[1] == 0:
    mark = 'o'
else:
    mark = 'x'  
    
plt.plot(x[0,1] , x[1,1],c='r', marker = mark)

if t[2] == 0:
    mark = 'o'
else:
    mark = 'x'
    plt.plot(x[0,2] , x[1,2],c='r',marker = mark)

if t[3] == 0:
    mark = 'o'
else:
    mark = 'x'
plt.plot(x[0,3] , x[1,3],c='r',marker = mark)

m = -(weights_0[2]/weights_0[1])/(weights_0[2]/weights_0[0])
b = -(weights_0[2]/weights_0[1])
m=m[0]
b=b[0]
ly=[]
for lx in np.arange(0, 1.1, .1):
    ly.append(m * lx - b)
plt.plot(np.arange(0, 1.1, .1),ly)
plt.show()

print ("Epocas =",epocas)

print ("Tiempo en segundos:" , timer)
