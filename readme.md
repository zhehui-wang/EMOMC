
# EMOMC

Welcome to the project of "Evolutionary Multi-Objective Model Compression for Deep Neural Networks" This project has been testd on the Python 3.7.4

## Reference Paper
This is the code for the paper "Evolutionary Multi-Objective Model Compression for Deep Neural Networks". The following is the bibtex.

    @article{wang2021evolutionary,
      title={Evolutionary multi-objective model compression for deep neural networks},
      author={Wang, Zhehui and Luo, Tao and Li, Miqing and Zhou, Joey Tianyi and Goh, Rick Siow Mong and Zhen, Liangli},
      journal={IEEE Computational Intelligence Magazine},
      volume={16},
      number={3},
      pages={10--21},
      year={2021},
      publisher={IEEE}
    }



## Including Your own Model

Models are included in the folder **./model**. If you use the exiting models, please skip this section. If you want to add your own models, just editor a new python file **./model/myModel.py**. 

In the python file, we import the following library, where we customize the conv layers and linear layers.

    import compression as R

Then, for each conv layer or linear layer you want to compress, please replace the original module with our module. 

    nn.Conv2d --> R.Conv2d
    nn.Linear --> R.Linear


## Pre-train and Pre-prune the Model

You need to pre-train and pre-prune the model before optimziation. As the cluster of checkpoints are huge, we do not provide checkpoints. Please run the following command to prepare the checkpoints.

    Python EMOMC.py --working_mode 0 --network_type 0 --dataset_type 0 
This working mode 0 will automatically prepare everything, it takes several hours to prepare VGG-16 on CIFAR10.
## Evolutionary Optimization
We offer two types of optimization. working mode 1 is for the trade-off between accuracy and energy.

    Python EMOMC.py --working_mode 1 --network_type 0 --dataset_type 0 

 Working mode 2 is for the trade-off between accuracy and model size.

    Python EMOMC.py --working_mode 2 --network_type 0 --dataset_type 0 

## Change of Parameters
You can use -- dataflow_type or --coding_type to change the types. For other parameters, please use -h to see the description!
| Paramter|  Descrption|
|--|--|
|working_mode | 0 for pretrain pruned model, 1 for energy optimizatoin, 2 for model size optimization |
|device_id |index of the graphic card |
 network_type          |  0 for LeNet5, 1 for Mobilenet, 2 for VGG16, 3 for ResNet, 4 for customized model
dataset_type  | 0 for MINST, 1 for CIFAR10
  dataflow_type |0 for  X,Y, 1 for IC,OC, 2 for Fx,Fy, 3 for X,Fx
coding_type | 0 for normal, 1 for CSR, 2 for COO
population_size |number of nodes per generation
number_generations|number of generations
size_constraint |upper bound for size
energy_constraint |upper bound for energy
pre_train_epochs|the number of epoches to train a model from scratch
pre_prune_upper_bound|the maximum pruning remaining amount'
pre_prune_lower_bound |the minimum pruning remaining amount
pre_prune_step |the step of pre-prune'
pre_prune_epochs |the number of epoches to fine-tune a pre-prund model
