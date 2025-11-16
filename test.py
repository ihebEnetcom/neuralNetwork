import numpy as np
from neuralNetwork import MLP
X = np.array([
    [0 ,0],
    [0 ,1],
    [1 ,0],
    [1 ,1],
    
])  
Y = np.array([
    [0],
    [1],
    [1],
    [0],
    
])  
mlp= MLP(2,2,1,X,Y,25)

for i in range(0,100000):
    mlp.train()
print("0 xor 0")
print(mlp.predict(np.array([0,0])))
print("1 xor 0")
print(mlp.predict(np.array([1,0])))
print("0 xor 1")
print(mlp.predict(np.array([0,1])))
print("1 xor 1")
print(mlp.predict(np.array([1,1])))