import numpy as np
from PolicyLearning import *

def f(h1,h2):
    return np.linalg.norm(h1-h2*h2)

class phi_cuadratico:
    def __init__(self,a,b,c):
        self.params = np.array([a,b,c])
    def __call__(self, x):
        return self.params[0]*x*x + self.params[1]*x + self.params[2]

phi_cuad = phi_cuadratico(1,0,0)

modelo = model_free_primal_dual(phi_cuad,f,0.0001,lr=0.000001,batch_size=50,epochs=50000)
data = np.random.randint(0,1000,size=(1000,2))
phi_learned = modelo.train(data)


print(phi_learned.params)

h = np.array(data[0])
sol = phi_learned(h)

print(phi_learned.params)
print(h,sol,f(sol,h))