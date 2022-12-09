import numpy as np
from PolicyLearning import *

def f(h1,h2):
    return np.linalg.norm(h1-h2)

class phi_lineal:
    def __init__(self,a,b):
        self.params = np.array([a,b])
    def __call__(self, x):
        return self.params[0]*x + self.params[1]

phi_lin = phi_lineal(28,37)

modelo = model_free_primal_dual(phi_lin,f,0.0001,lr=0.005,batch_size=50,epochs=50000)
data = np.random.randint(0,50,size=(1000,2))
phi_learned = modelo.train(data)


print(phi_learned.params)

h = np.array([3,2])
sol = phi_lin(h)

print(h,sol,f(sol,h))