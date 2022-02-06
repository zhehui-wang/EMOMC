import numpy as np
import math
import torch
import torch.nn as nn
import config as C



#unit is Kbyte
def calculate_size(coding_type):

    #fetch compression configuration
    prune_amount, quantize_bit = C.get_config() 
 
    #fetch the conv and lienar layer lists
    layers = C.get_layer()
    layer_dim = len(layers)

    #find the size of weights in each layer
    size_table = [0] * layer_dim
    for i in range (layer_dim):
        if (layers[i].type == 0):
            size_table[i] = (layers[i].kernel_size[0] * layers[i].kernel_size[1]) * (layers[i].in_channels * layers[i].out_channels / layers[i].groups)
        elif(layers[i].type == 1):
            size_table[i] = layers[i].in_features * layers[i].out_features
   
    #accumulate total size
    total_size = 0
    for i in range (layer_dim):
        if (coding_type == 0): 
            #assume 0 value weight are also transmmiting and stored
            total_size += size_table[i]* round(quantize_bit[i])
        elif (coding_type == 1):
            # extra 3bits is used to indicate the location of next non-zero element
            total_size += size_table[i]* round(quantize_bit[i]+3)* prune_amount 
        elif (coding_type == 2):
            # extra index to indicate the location of non-zero elements
            total_size += size_table[i]* round(quantize_bit[i] + math.log(size_table[i], 2)) * prune_amount 

    #changes to KBytes
    total_size /= (1024*8)

    return total_size


# pre-calculate resource for energy consumption
def calculate_resource(dataflow_type):  
   
    #fetch the conv and lienar layer lists
    layers = C.get_layer()
    layer_dim = len(layers)

    #0 for mac, 1 for ifmap, 2 for weight, 3 for ofmap_output, 4 for ofmap_input
    resource_table = [[0 for i in range(layer_dim)] for j in range(5)]

    for i in range(layer_dim):
        #for conv layer
        if (layers[i].type == 0):
            #get the group value 
            groups = layers[i].groups
        
            #find the number of macs
            resource_table[0][i] = layers[i].kernel_size[0]*layers[i].kernel_size[1]*layers[i].ofmap_size
            resource_table[0][i] = resource_table[0][i]*(layers[i].in_channels/groups)*(layers[i].out_channels/groups)*groups
            #0 for X|Y
            if (dataflow_type == 0):
                resource_table[1][i] = resource_table[0][i]/ 1
                resource_table[2][i] = resource_table[0][i]/ (layers[i].ofmap_size)
                resource_table[3][i] = resource_table[0][i]/ (layers[i].kernel_size[0]*layers[i].kernel_size[1]*(layers[i].in_channels/groups))
                resource_table[4][i] = 0
            #1 for IC|OC
            if (dataflow_type == 1):
                resource_table[1][i] = resource_table[0][i]/ (layers[i].out_channels/groups)
                resource_table[2][i] = resource_table[0][i]/ 1
                resource_table[3][i] = resource_table[0][i]/ (layers[i].in_channels/groups)
                resource_table[4][i] = resource_table[3][i]
            #2 for Fx|Fy
            if (dataflow_type == 2):
                resource_table[1][i] = resource_table[0][i]/ 1
                resource_table[2][i] = resource_table[0][i]/ (layers[i].ofmap_size)
                resource_table[3][i] = resource_table[0][i]/ (layers[i].kernel_size[0]*layers[i].kernel_size[1])
                resource_table[4][i] = resource_table[3][i]
            #3 for X|Fx
            if (dataflow_type == 3):
                resource_table[1][i] = resource_table[0][i]/ (layers[i].kernel_size[0])
                resource_table[2][i] = resource_table[0][i]/ (layers[i].ofmap_size**0.5)
                resource_table[3][i] = resource_table[0][i]/ (layers[i].kernel_size[0])
                resource_table[4][i] = resource_table[3][i]                                
        
        #for linear layer
        elif(layers[i].type == 1):           
            resource_table[0][i] = layers[i].in_features*layers[i].out_features
            # default on output stationary
            resource_table[1][i] = resource_table[0][i]/ 1
            resource_table[2][i] = resource_table[0][i]/ 1
            resource_table[3][i] = layers[i].out_features
            resource_table[4][i] = 0
 
    return resource_table

 

#unit is 0.01mJ
def calculate_energy(dataflow_type):

    #pre-calculate resource
    resource_table = calculate_resource(dataflow_type)

    #fetch the conv and lienar layer lists
    layers = C.get_layer()
    layer_dim = len(layers)

    #fetch compression configuration
    prune_amount, quantize_bit = C.get_config()

    #energy parameters
    eff_mac, eff_read, eff_write = 0.102, 0.728, 0.447

    #0 for mac, 1 for ifmap, 2 for weight, 3 for ofmap_output, 4 for ofmap_input
    energy = [0.0] * 5    

    #calculate the energy
    for i in range(layer_dim):  
        #find the effective mac operations
        energy[0] += eff_mac * resource_table[0][i] * prune_amount * (round(quantize_bit[i])+1)
        # all the weight with 0 values have to be read
        energy[2] += eff_read * resource_table[2][i] * (round(quantize_bit[i])+1) *(1/16)  
        #for conv layer        
        if (layers[i].type == 0):
            groups = layers[i].groups
            #0 for X|Y
            if (dataflow_type == 0):
                energy[1] += eff_read * resource_table[1][i] * prune_amount
                energy[3] += eff_write * resource_table[3][i]
                energy[4] += eff_read * resource_table[4][i]          
            #1 for IC|OC
            if (dataflow_type == 1):
                if layers[i].out_channels/groups >24:
                    energy[1] += eff_read * resource_table[1][i]
                else :                    
                    energy[1] += eff_read * resource_table[1][i] * (1-(1-prune_amount)**(layers[i].out_channels/groups))      
                if layers[i].in_channels/groups >24:
                    energy[3] += eff_write * resource_table[3][i]
                    energy[4] += eff_read * resource_table[4][i]  
                else:
                    energy[3] += eff_write * resource_table[3][i] * (1-(1-prune_amount)**(layers[i].in_channels/groups))      
                    energy[4] += eff_read * resource_table[4][i] * (1-(1-prune_amount)**(layers[i].in_channels/groups))                    
            #2 for Fx|Fy
            if (dataflow_type == 2):
                energy[1] += eff_read * resource_table[1][i] * prune_amount
                energy[3] += eff_write * resource_table[3][i] * (1-(1-prune_amount)**(layers[i].kernel_size[0]*layers[i].kernel_size[1]))
                energy[4] += eff_read * resource_table[4][i] * (1-(1-prune_amount)**(layers[i].kernel_size[0]*layers[i].kernel_size[1]))      
            #3 for X|Fx
            if (dataflow_type == 3):
                energy[1] += eff_read * resource_table[1][i] * (1-(1-prune_amount)**layers[i].kernel_size[0])
                energy[3] += eff_write * resource_table[3][i] * (1-(1-prune_amount)**layers[i].kernel_size[0])
                energy[4] += eff_read * resource_table[4][i] * (1-(1-prune_amount)**layers[i].kernel_size[0])            

        #for linear layers, only for output stationary mode
        elif(layers[i].type == 1):
            energy[1] += eff_read * resource_table[1][i] * prune_amount
            energy[3] += eff_write * resource_table[3][i]
       
    # scale the energy by 1E-6
    result_scale=1000000
    total_energy = sum(energy)/result_scale  
    return total_energy
