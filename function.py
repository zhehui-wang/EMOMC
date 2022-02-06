import numpy as np
import math
import random
import torch
import torch.nn as nn

import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys  
import argparse
import config as C
import evaluation as E
import models.MobileNet as MobileNet
import models.LeNet as LeNet
import models.VGG as VGG
import models.ResNet as ResNet
import models.myModel as myModel

from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.distributions import Normal
from collections import namedtuple


parser = argparse.ArgumentParser(description='function')

# !important, for basic configurations
parser.add_argument('--device_id', type = int, default=0,  help='index of the graphic card')
parser.add_argument('--working_mode', type = int, default=1,  help='0 for pretrain pruned model, 1 for energy optimizatoin, 2 for model size optimization')

# !important, for basic configuration of the model
parser.add_argument('--network_type', type = int, default=0,  help='0 for LeNet5, 1 for Mobilenet, 2 for VGG16, 3 for ResNet-18, 4 for customized model')
parser.add_argument('--dataset_type', type = int, default=0,  help='0 for MINST, 1 for CIFAR10')
parser.add_argument('--dataflow_type', type = int, default=0,  help='0 for  X|Y, 1 for IC|OC, 2 for Fx|Fy, 3 for X|Fx')
parser.add_argument('--coding_type', type = int, default=1,  help='0 for normal, 1 for CSR, 2 for COO')

# for Evolutionary algorithm
parser.add_argument('--population_size', type = int, default=20,  help='number of nodes per generation')
parser.add_argument('--number_generations', type = int, default=120,  help='number of generations')
parser.add_argument('--size_constraint', type = int, default=10000,  help='upper bound for size')
parser.add_argument('--energy_constraint', type = int, default=10000,  help='upper bound for energy')

# for pre traing from scratch (step 1) and pre prune models (step 2).
parser.add_argument('--pre_train_epochs', type = int, default=300,  help='the number of epoches to train a model from scratch')
parser.add_argument('--pre_prune_upper_bound', type = int, default=80,  help='the maximum pruning remaining amount')
parser.add_argument('--pre_prune_lower_bound', type = int, default=20,  help='the minimum pruning remaining amount')
parser.add_argument('--pre_prune_step', type = int, default=1,  help='the step of pre-prune')
parser.add_argument('--pre_prune_epochs', type = int, default=32,  help='the number of epoches to fine-tune a pre-prund model')
 
args = parser.parse_args()

print("Device ID", args.device_id)
print("Working mode", args.working_mode)
print("Network", args.network_type)
print("Dataset", args.dataset_type)
print("Dataflow type", args.dataflow_type)
print("Coding type", args.coding_type)



# set dataset
if (args.dataset_type == 0):
    data_train = MNIST('./data/mnist', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    data_test = MNIST('./data/mnist', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=2048, num_workers=8)
elif (args.dataset_type == 1):
    data_train = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
    data_test = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
    data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=1024, shuffle=False, num_workers=8)


#setting the target network
if (args.network_type == 0): net = LeNet.LeNet5()
elif (args.network_type == 1): net = MobileNet.MobileNet()
elif (args.network_type == 2): net = VGG.VGGs(16)
elif (args.network_type == 3): net = ResNet.Resnet(18)
elif (args.network_type == 4): net = myModel.myModel()


#define net property
net.cuda(args.device_id) 
criterion = nn.CrossEntropyLoss()
if (args.network_type == 0):  optimizer = optim.Adam(net.parameters(), lr=1e-2)
if (args.network_type == 1):  optimizer = optim.Adam(net.parameters(), lr=1e-2)
if (args.network_type == 2):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)
if (args.network_type == 3):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)
if (args.network_type == 4):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):  
    if (args.network_type == 0):  base_lr = 0.01
    if (args.network_type == 1):  base_lr = 0.01
    if (args.network_type == 2):  base_lr = 0.05
    if (args.network_type == 3):  base_lr = 0.05
    if (args.network_type == 4):  base_lr = 0.05
   
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):      
        optimizer.zero_grad()
        #send to GPU
        images, labels=images.cuda(args.device_id), labels.cuda(args.device_id)
        images, labels=Variable(images), Variable(labels)
        C.set_is_quantized(0)
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    if (args.working_mode == 0): print('Train - Epoch %d, Loss: %f' % (epoch, loss.detach().cpu().item()))

def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            #send to GPU
            images, labels=images.cuda(args.device_id), labels.cuda(args.device_id)
            images, labels=Variable(images), Variable(labels)
            C.set_is_quantized(0)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(data_test)
    if (args.working_mode == 0): print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    return float(total_correct) / len(data_test)

def test_half():
    net.eval()
    net.half()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            #send to GPU
            images, labels=images.cuda(args.device_id).half(), labels.cuda(args.device_id)
            images, labels=Variable(images), Variable(labels)
            C.set_is_quantized(1)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(data_test)
    net.float()
    return float(total_correct) / len(data_test)


def pre_train():
    for epoch in range(args.pre_train_epochs):
        adjust_learning_rate(optimizer, epoch)
        train(epoch)
        test()        
    torch.save({'network': net.state_dict(), 'optimizer': optimizer.state_dict()}, './checkpoints/pre_trained_'+str(args.network_type)+'_'+str(args.dataset_type)+'.pth')
    print("pre-train the model successful!")
    return

def pre_prune(pruning_remaining_amount):
 
    #fetch the pre-trained models
    checkpoint = torch.load('./checkpoints/pre_trained_'+str(args.network_type)+'_'+str(args.dataset_type)+'.pth')  
    net.load_state_dict(checkpoint['network'])  
    optimizer.load_state_dict(checkpoint['optimizer'])

    #pruning amount of each step
    delta_prune = pow(pruning_remaining_amount/100, (1/args.pre_prune_epochs))   

    #fine-tune the model
    for step in range(args.pre_prune_epochs): 
        C.set_config_prune(prune_amount_in = delta_prune**(step+1))
        train(step)   
        test()
    torch.save({'network': net.state_dict(), 'optimizer': optimizer.state_dict()}, './checkpoints/pre_trained_'+str(args.network_type)+'_'+str(args.dataset_type)+'_'+str(pruning_remaining_amount)+'.pth')
    print("pre-prune the model with pruning remaining amount: "+ str(pruning_remaining_amount) +" successful!" )
    return

def sample_point(target_prune, target_quantize):

    #fetch the pre-pruned models
    checkpoint = torch.load('./checkpoints/pre_trained_'+str(args.network_type)+'_'+str(args.dataset_type)+'_'+str(target_prune)+'.pth')  
    net.load_state_dict(checkpoint['network'])  
    optimizer.load_state_dict(checkpoint['optimizer'])

    C.set_config(target_prune/100, target_quantize)

    #print the info
    print('Pruning remaining amount: %.3f' %(target_prune/100)) 
    print('Quantization bits: ', end = '\t')
    for i in range (len(target_quantize)): print('%.1f' %target_quantize[i], end= '\t')
    print(" ")    

    #fetch results
    accuracy = test_half()
    energy = E.calculate_energy(args.dataflow_type)
    size = E.calculate_size(args.coding_type)
    print('Accuracy = %.3f' % accuracy, "\t", 'Energy = %.2f' %energy, "\t", 'Size = %.1f' %size, "\t")
             
    return accuracy, energy, size






