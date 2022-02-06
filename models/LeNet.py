import numpy as np
import math
import random
import torch
import torch.nn as nn 
import compression as R
 

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #conv layers
        self.conv1 = R.Conv2d(1, 6, 5)
        self.conv2 = R.Conv2d(6, 16, 5)
        self.conv3 = R.Conv2d(16, 120, 5)
        #fc layer
        self.linear1 = R.Linear(120, 84)
        self.linear2 = R.Linear(84, 10)
        #pool layer
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 
        #relu layer
        self.relu1 = nn.ReLU() 
        self.relu2 = nn.ReLU() 
        self.relu3 = nn.ReLU() 
        self.relu4 = nn.ReLU() 
        #softmax layer
        self.softmax = nn.LogSoftmax(dim=-1)          
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu4(self.linear1(x))        
        x = self.linear2(x)
        x = self.softmax(x)
        return x