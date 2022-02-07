#pre-process VGG16 on CIFAR-10
python EMOMC.py --working_mode 0 --network_type 2 --dataset_type 1 
#trade-off between accuracy and energy on dataflow type X|Fx
python EMOMC.py --working_mode 1 --network_type 2 --dataset_type 1 --dataflow_type 3
