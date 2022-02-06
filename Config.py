import numpy as np
import math
import random
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys  


prune_amount = 1
quantize_bit = []

id_counter = 0

is_quantized = 1

layer_list = []
 

def set_config(prune_amount_in, quantize_bit_in):
    
    global prune_amount, quantize_bit

    prune_amount, quantize_bit = prune_amount_in, quantize_bit_in

    return  


def set_config_prune(prune_amount_in):
    
    global prune_amount

    prune_amount = prune_amount_in
 
    return     

def get_config():

    global prune_amount, quantize_bit
    
    return prune_amount, quantize_bit
  

def set_counter():

    global id_counter

    id_counter += 1

    return

def get_counter():

    global id_counter

    return id_counter 



def set_is_quantized(flag):

    global is_quantized
    
    is_quantized = flag

    return

def get_is_quantized():

    global is_quantized

    return is_quantized

def set_layer(layer):

    global layer_list  

    layer_list.append(layer)

    return

def get_layer():

    global layer_list  

    return layer_list


