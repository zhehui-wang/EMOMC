import numpy as np
import math
import torch
import torch.nn as nn
import config as C
import torch.nn.functional as F

class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.ofmap_size = 0
        #ofmap_size is the size of the feature map, which will be calulated during inference
        self.type = 0   
        self.id = C.get_counter()
        C.set_counter()
        C.set_layer(self)
 
    #overwrite the forward function for pruning
    def forward(self, input):
        prune_amount, quantize_bit = C.get_config()
        is_quantized = C.get_is_quantized()
        if is_quantized == 0:
            self.weight_pruned = prune_tensor(self.weight, prune_amount)
            return F.conv2d(input, self.weight_pruned, self.bias, self.stride, self.padding, self.dilation, self.groups)  
        elif is_quantized == 1:    
            self.weight_pruned = prune_tensor(self.weight, prune_amount)
            self.weight_quantized = quantize_tensor (self.weight_pruned, quantize_bit[self.id])
            #after we do inference, we get the ofmap size.
            output = F.conv2d(input, self.weight_quantized, self.bias, self.stride, self.padding, self.dilation, self.groups)    
            self.ofmap_size = output.shape[2]*output.shape[3]
            return output

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.type = 1
        self.id = C.get_counter()
        C.set_counter()
        C.set_layer(self)

    #overwrite the forward function for pruning
    def forward(self, input):
        prune_amount, quantize_bit = C.get_config()
        is_quantized = C.get_is_quantized()
        if is_quantized == 0:
            self.weight_pruned = prune_tensor(self.weight, prune_amount)
            return F.linear(input, self.weight_pruned, self.bias)
        elif is_quantized == 1:    
            self.weight_pruned = prune_tensor(self.weight, prune_amount)
            self.weight_quantized = quantize_tensor (self.weight_pruned, quantize_bit[self.id])
            return F.linear(input, self.weight_quantized, self.bias)

def prune_tensor(tensor_pruned, prune_amount):
        #create a mask
        mask=torch.ones_like(tensor_pruned)        
        #let the first several number of parameters to be 0
        prunt_number=int(round(tensor_pruned.nelement()*(1-prune_amount)))
        pruned_index = torch.abs(tensor_pruned).view(-1).argsort()[:prunt_number]
        mask.view(-1)[pruned_index] = 0
        tensor_pruned = mask.to(dtype=tensor_pruned.dtype)*tensor_pruned
        return tensor_pruned

def quantize_tensor(tensor_quantized, quantize_bit):
        #find the spacing
        tensor_max, tensor_min = torch.max(tensor_quantized), torch.min(tensor_quantized)
        scale = max(abs(tensor_max.item()), abs(tensor_min.item()))
        spacing = scale/(2**(quantize_bit-1)-1)
        #quantize the pruned weight
        tensor_quantized = torch.round(tensor_quantized/spacing)*spacing
        tensor_quantized = tensor_quantized.half()
        return tensor_quantized
