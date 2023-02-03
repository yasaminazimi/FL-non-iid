import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Device config.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters
learning_rate = 0.001


class CNN(nn.Module):
    def __intit__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5) 
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 128)
        self.fc3 = nn.Linear(256,10)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def accuracy(model, dataset):
    
    correct=0
    
    for features,labels in iter(dataset):
        predictions= model(features)
        _,predicted=predictions.max(1,keepdim=True)
        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
        
    accuracy = 100*correct/len(dataset.dataset)
        
    return accuracy
