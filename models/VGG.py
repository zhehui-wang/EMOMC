import numpy as np
import math
import random
import torch
import torch.nn as nn 
import compression as R




cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = R.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [R.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
 
def VGGs(type_id):

    if(type_id==11):  net = VGG('VGG11')
    if(type_id==13):  net = VGG('VGG13')
    if(type_id==16):  net = VGG('VGG16')
    if(type_id==19):  net = VGG('VGG19')
    return net






# class VGG16(nn.Module):
#     def __init__(self):
#         super(VGG16, self).__init__()
#         #conv layers
#         #stack 0
#         self.conv_list = nn.ModuleList([R.Conv2dCompress(3, 64, 3, stride=1, padding=1)])
#         self.conv_list.append(R.Conv2dCompress(64, 64, 3, stride=1, padding=1))
#         #stack 1
#         self.conv_list.append(R.Conv2dCompress(64, 128, 3, stride=1, padding=1))
#         self.conv_list.append(R.Conv2dCompress(128, 128, 3, stride=1, padding=1))    
#         #stack 2
#         self.conv_list.append(R.Conv2dCompress(128, 256, 3, stride=1, padding=1))
#         self.conv_list.append(R.Conv2dCompress(256, 256, 3, stride=1, padding=1))
#         self.conv_list.append(R.Conv2dCompress(256, 256, 3, stride=1, padding=1))
#         #stack 3
#         self.conv_list.append(R.Conv2dCompress(512, 512, 3, stride=1, padding=1))  
#         #stack 4
#         self.conv_list.append(R.Conv2dCompress(512, 512, 3, stride=1, padding=1))
#         self.conv_list.append(R.Conv2dCompress(512, 512, 3, stride=1, padding=1))
#         self.conv_list.append(R.Conv2dCompress(512, 512, 3, stride=1, padding=1))  
#         #initlaize the weights of the conv layers        
#         for i in range (13):        
#             n = self.conv_list[i].kernel_size[0] * self.conv_list[i].kernel_size[1] * self.conv_list[i].out_channels
#             self.conv_list[i].weight.data.normal_(0, math.sqrt(2. / n))
#             self.conv_list[i].bias.data.zero_()
#         #fc layer
#         self.fulc_list = nn.ModuleList ([R.LinearCompress(512, 512)])
#         self.fulc_list.append(R.LinearCompress(512, 512))
#         self.fulc_list.append(R.LinearCompress(512, 10))
#         #pool layer
#         self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for i in range(5)])      
#         #relu layer  
#         self.relu_list = nn.ModuleList([nn.ReLU(inplace = True) for i in range(15)])  
#         #dropout layer
#         self.drop_list = nn.ModuleList([nn.Dropout() for i in range(2)])
#     #forward functions for training              
#     def forward(self, x):    
#         for i in range(13):
#             x = self.conv_list[i](x)  
#             x = self.relu_list[i](x)
#             if i == 1:    x = self.pool_list[0](x)
#             elif i == 3:  x = self.pool_list[1](x)
#             elif i == 6:  x = self.pool_list[2](x)
#             elif i == 9:  x = self.pool_list[3](x)
#             elif i == 12: x = self.pool_list[4](x)            
#         x = x.view(-1, 512)
#         for i in range(2):
#             x = self.drop_list[i](x)
#             x = self.fulc_list[i](x)
#             x = self.relu_list[13+i](x)
#         x = self.fulc_list[2](x)
#         return x

