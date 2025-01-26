
k = 12


import torch
import random
import os
import sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import copy

# setting path
# your path to code


from layer_insertion_oo import training_with_one_LI
from training_oo import train_classical
from nets import build_vgg_baseline, extend_VGG
from data_preparation import prep_cifar10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed
s=1
random.seed(s)
np.random.seed(s)
torch.manual_seed(s)

trainloader, testloader = prep_cifar10()


#################################################################################

epochs1 = [10,190]
epochs2 = [20,180]
epochs3 = [30,170]
epochs4 = [40,160]
epochs5 = [50,150]
epochs6 = [60,140]
epochs7 = [70,130]
epochs8 = [80,120]
epochs_class = sum(epochs1)
BN=False
lr_init = 0.01
lr_class_small = lr_init
lr_class_big = lr_init


lr_args = {'step_size': 10000, 
           'gamma': 0.5}

#################################################################################

m=[64,128,256]

model_small = build_vgg_baseline(BN,m=m)
init_vec1 = torch.nn.utils.parameters_to_vector(model_small.parameters())
init_vec2 = copy.deepcopy(init_vec1.data)
init_vec3 = copy.deepcopy(init_vec1.data)
init_vec4 = copy.deepcopy(init_vec1.data)
init_vec5 = copy.deepcopy(init_vec1.data)
init_vec6 = copy.deepcopy(init_vec1.data)
init_vec7 = copy.deepcopy(init_vec1.data)
init_vec8 = copy.deepcopy(init_vec1.data)
init_vec9 = copy.deepcopy(init_vec1.data)
init_vec10 = copy.deepcopy(init_vec1.data)

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

path = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}'

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
            epochs=epochs1, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec2)
        
        path1 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_1_{init}.json'
        Res1.save_to_json(path1)

    # absmax
    if T2:
        print('T2:')
        Res2 = training_with_one_LI(
            epochs=epochs2, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec3)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_2_{init}.json'
        Res2.save_to_json(path2)

    # absmax
    if T3:
        print('T3:')
        Res3 = training_with_one_LI(
            epochs=epochs3, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec4)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_3_{init}.json'
        Res3.save_to_json(path2)

    # absmax
    if T4:
        print('T4:')
        Res4 = training_with_one_LI(
            epochs=epochs4, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec5)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_4_{init}.json'
        Res4.save_to_json(path2)

    # absmax
    if T5:
        print('T5:')
        Res5 = training_with_one_LI(
            epochs=epochs5, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec6)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_5_{init}.json'
        Res5.save_to_json(path2)

    # absmax
    if T6:
        print('T6:')
        Res6 = training_with_one_LI(
            epochs=epochs6, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec7)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_6_{init}.json'
        Res6.save_to_json(path2)

    # absmax
    if T7:
        print('T7:')
        Res7 = training_with_one_LI(
            epochs=epochs7, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec8)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_7_{init}.json'
        Res7.save_to_json(path2)

    # absmax
    if T8:
        print('T8:')
        Res8 = training_with_one_LI(
            epochs=epochs8, traindataloader=trainloader, testdataloader=testloader,BN=BN,
            optimizer_type='SGD', lr_init=lr_init, mode = 'threshold_rel', stopping_criterion=None,
            lrschedule_type='StepLR', lrscheduler_args=lr_args, 
            decrease_lr_after_li=1.,save_grad_norms=True, init=init_vec9)
        
        path2 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_8_{init}.json'
        Res8.save_to_json(path2)

    # train classical small
    if T9:
        print('T9:')
        model_class_small = build_vgg_baseline(BN,m=m)
        torch.nn.utils.vector_to_parameters(init_vec10, model_class_small.parameters())
        optimizer_small = torch.optim.SGD(model_class_small.parameters(),lr_class_small,
                                          momentum = 0.9, weight_decay=5e-3)
        scheduler_small = torch.optim.lr_scheduler.StepLR(
            optimizer_small, step_size=lr_args['step_size'], gamma=lr_args['gamma'])
        
        Res9 = train_classical(model_class_small,trainloader,testloader,optimizer_small,epochs_class,scheduler_small,
                                                    save_grad_norms=True)
        path3 = f'yourpath/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_9_{init}.json'
        Res9.save_to_json(path3)



