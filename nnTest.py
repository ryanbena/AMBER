import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("GPU is available:")
print(torch.cuda.is_available())
print("Device count:")
print(torch.cuda.device_count())

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2,5),
            nn.Tanh(),
            nn.Linear(5,5),
            nn.Tanh(),
            nn.Linear(5,1)
        )
    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

myNN = NeuralNetwork()
#optimizer = optim.SGD(myNN.parameters(), lr=0.001, momentum=0.99)
optimizer = optim.Adam(myNN.parameters(), lr=0.01, betas=(0.99, 0.999))
loss_fn = nn.MSELoss()

xmax = 100
ymax = 100
def target_fn(x,y):
    center = [0.5,0.5]
    radius = 0.25
    sqDistance = (x/xmax - center[0])**2 + (y/ymax - center[1])**2
    target = sqDistance - radius**2
    return target

running_loss = 0.0
for epoch in range(10002):
    
    optimizer.zero_grad()

    batch = 100
    batch_inputs = torch.zeros(batch,2) 
    batch_target = torch.zeros(batch,1)
    
    for k in range(batch):
        batch_inputs[k,0] = torch.randint(0,xmax,(1,1))
        batch_inputs[k,1] = torch.randint(0,ymax,(1,1))
        batch_target[k] = target_fn(batch_inputs[k,0],batch_inputs[k,1])

    batch_sol = myNN(batch_inputs)
    loss = loss_fn(batch_sol,batch_target)
    loss.backward()
    optimizer.step()

    running_loss += loss
    
    if not epoch.__mod__(100):
        print(running_loss/100)
        print(epoch)
        running_loss = 0.0
    
    if not (epoch+1).__mod__(10000):
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        learned_grid = torch.zeros(xmax,ymax)
        for i in range(xmax):
            for j in range(ymax):
                learned_grid[i,j] = myNN(torch.Tensor([i, j]))
        grid = learned_grid.detach().numpy()
        x = torch.arange(0, xmax, 1)
        y = torch.arange(0, ymax, 1)
        X, Y = torch.meshgrid(x,y)
        Z = grid.reshape(X.shape)
        surface = ax.plot_surface(X,Y,Z)
        plt.show()