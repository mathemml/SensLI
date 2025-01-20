k = 1


import torch
import random
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import copy

# setting path
sys.path.append('/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/code')


from layer_insertion_oo import training_with_one_LI
from training_oo import train_classical
from nets import build_vgg_baseline, extend_VGG, extend_VGG_new
from data_preparation import prep_cifar10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

trainloader, testloader = prep_cifar10()


#################################################################################
max_LIs = 3

epochs = [50,50,50,50]
epochs_class = sum(epochs)
BN=False
lr_init = 0.01
lr_class_small = lr_init
lr_class_big = lr_init




lr_args = {'step_size': 10, # patience
           'gamma': 0.5}    # lr scaling

#################################################################################
m = [64,128,256]


model_small = build_vgg_baseline(BN=BN,m=m)
init_vec1 = torch.nn.utils.parameters_to_vector(model_small.parameters())
init_vec2 = copy.deepcopy(init_vec1.data)
init_vec3 = copy.deepcopy(init_vec1.data)
init_vec4 = copy.deepcopy(init_vec1.data)
init_vec5 = copy.deepcopy(init_vec1.data)
init_vec6 = copy.deepcopy(init_vec1.data)
init_vec7 = copy.deepcopy(init_vec1.data)
init_vec8 = copy.deepcopy(init_vec1.data)

################################################################################
# determine which trainings are run
T1 = True
T2 = True
T3 = True
T4 = True
T5 = True
T6 = True
T7 = True
T8 = True
T9 = True

# define no of training run instances

no_of_initializations = 1

path = f'/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/results_data/Exp{k}'

# check if repo already exists
if os.path.exists(path):
    print(f"Repo with path {path} exists already!")
    quit()

# create repo for json files of this experiment:
os.mkdir(path)


# check if repo was created
if os.path.exists(path):
    print("Repo created successfully!")
else:
    print("Repo could not be created :/")



for init in range(no_of_initializations):
    print(f'run number {init}:')

    # absmax
    if T1:
        print('T1:')

        Res1 = training_with_one_LI(
            epochs=epochs, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='Adam', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='ReduceLROnPlateau', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec2, sens_norm='all',max_LIs=max_LIs)
        
        path1 = f'/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_1_{init}.json'
        Res1.save_to_json(path1)
        positions = Res1.positions

    



     # train classical big
    if T4:
        print('T4:')
    
        model_class_big = extend_VGG(position=positions[0],BN=BN)
        for i in range(len(positions)-1):
            model_class_big = extend_VGG_new(model_class_big, position=positions[i+1],BN=BN)
        
        #optimizer_big = torch.optim.SGD(model_class_big.parameters(),lr_class_big,momentum = 0.9, weight_decay=5e-4)
        optimizer_big = torch.optim.Adam(model_class_big.parameters(),lr_class_big)
        scheduler_big = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_big, mode='max', factor=lr_args['gamma'], patience=lr_args['step_size'], verbose=False)
        
        Res4 = train_classical(model_class_big,trainloader,testloader,optimizer_big,epochs_class,scheduler_big,save_grad_norms=True)
        path4 = f'/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_4_{init}.json'
        Res4.save_to_json(path4)

    # train classical small
    if T3:
        print('T3:')
        model_class_small = build_vgg_baseline(BN,m=m)
        torch.nn.utils.vector_to_parameters(init_vec4, model_class_small.parameters())
        #optimizer_small = torch.optim.SGD(model_class_small.parameters(),lr_class_small,momentum = 0.9, weight_decay=5e-4)
        optimizer_small = torch.optim.Adam(model_class_small.parameters(),lr_class_small)
        scheduler_small = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_small, mode='max', factor=lr_args['gamma'], patience=lr_args['step_size'], verbose=False)
        
        Res3 = train_classical(model_class_small,trainloader,testloader,optimizer_small,epochs_class,scheduler_small,
                                                    save_grad_norms=True)
        path3 = f'/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_3_{init}.json'
        Res3.save_to_json(path3)
