import torch
import torch.nn as nn
import torch.nn.functional as F



class TargetNetwork(nn.Module):
    def __init__ (self, inputs, outputs, num_params, num_layers=1): 

        self.layer1 = nn.Linear(inputs, num_params)
        #self.activation = F.relu


    def initHidden(self): 
        return torch.zeros(1, self.hidden_size)


    def forward(self, x): 
        x = F.relu(self.layer1(x))
        return x
