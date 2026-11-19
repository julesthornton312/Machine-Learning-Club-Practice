import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN = nn.Sequential()
        self.CNN.append(nn.Conv2d( 1, 32, 3, padding=1))
        self.CNN.append(nn.ReLU())
        self.CNN.append(nn.MaxPool2d(kernel_size = 2))
        self.CNN.append(nn.Conv2d(32, 64, kernel_size = 3, padding = 1))
        self.CNN.append(nn.ReLU())
        self.CNN.append(nn.MaxPool2d(kernel_size = 2))

        self.MLP = nn.Sequential()
        self.MLP.append(nn.Linear(3136, 128))
        self.MLP.append(nn.ReLU())
        self.MLP.append(nn.Linear(128, 10))
        

    def forward(self, input):
        x=self.CNN(input).flatten(1)
        return self.MLP(x)



    
