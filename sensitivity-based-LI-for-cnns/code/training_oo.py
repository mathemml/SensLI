from typing import List, Tuple, Callable
import json

import torch
import model_selection_oo 
from train_and_test_ import train

class VGGExperiment:
    def __init__(self, model = None,
                 trainloader=None, 
                 testloader=None, 
                 optimizer = None, 
                 no_epochs = 1, 
                 scheduler=None, 
                 stopping_criterion = None, 
                 device=None) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.no_epochs = no_epochs
        self.scheduler = scheduler
        self.stopping_criterion = stopping_criterion


class VGGExperimentLI:
    def __init__(self, trainloader=None, 
                 testloader=None,
                 mode = 'max', 
                 optimizer = None, 
                 epochs = [1,1], 
                 scheduler=None, 
                 stopping_criterion = None,
                 decrease_lr_afterli = 1., 
                 device=None) -> None:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = model_selection_oo.VGGExtendableModel().model
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.no_epochs = epochs
        self.scheduler = scheduler
        self.stopping_criterion = stopping_criterion
        self.mode = mode
        self.decrease_lr_afterli = decrease_lr_afterli


class ResultClassical:
    def __init__(self) -> None:
        self.losses=[]
        self.testaccs = []
        self.trainaccs = []
        self.times = []
        self.grad_norms = []


    def add_losses(self, losses):
        self.losses += losses
    
    def add_testaccs(self, accs):
        self.testaccs += accs

    def add_trainaccs(self, accs):
        self.trainaccs += accs
    
    def add_times(self, times):
        self.times += times

    def add_grad_norms(self, grad_norms):
        self.grad_norms += grad_norms

    def add_all(self, losses, testaccs,train_accs, times, grad_norms):
        self.add_losses(losses)
        self.add_testaccs(testaccs)
        self.add_trainaccs(train_accs)
        self.add_times(times)
        self.add_grad_norms(grad_norms)

    def save_to_json(self, path):
        with open(path, 'w') as file:
            json.dump(self.__dict__, file)





class ResultLI(ResultClassical):
    def __init__(self) -> None:
        super().__init__()
        self.positions = []
        self.insertion_times = []
        self.insertion_epochs = []
        self.sensitivities = []
        self.sensitivities_all = []

    def add_all(self, losses, testaccs,train_accs, times, grad_norms):
        return super().add_all(losses, testaccs,train_accs, times, grad_norms)
    
    def add_times(self, times):
        return super().add_times(times)
    
    def add_losses(self, losses):
        return super().add_losses(losses)
    
    def add_testaccs(self, accs):
        return super().add_testaccs(accs)
    
    def add_trainaccs(self, accs):
        return super().add_trainaccs(accs)
    
    def add_grad_norms(self, grad_norms):
        return super().add_grad_norms(grad_norms)
    
    def add_position(self, pos):
        self.positions.append(pos)

    def add_insertion_time(self, time):
        self.insertion_times.append(time)

    def add_insertion_epoch(self, epoch):
        self.insertion_epochs.append(epoch)

    def add_sensitivities(self, sens):
        self.sensitivities.append(sens)

    def add_sensitivities_all(self, sens_all):
        self.sensitivities_all.append(sens_all)

    def add_ext_full_grad(self, g):
        self.ext_full_grad = g

    def add_grads_of_ext_model(self, l):
        self.ext_model_grads = l


def json_to_obj(path, obj='LI'):
    if obj!='LI' and obj!='Classical':
        return NotImplementedError
    if obj=='LI':
        Res = ResultLI()
    if obj=='Classical':
        Res = ResultClassical()
    f = open(path)
    dict_from_json = json.load(f)
    Res.__dict__ = dict_from_json
    return Res



# Test = ResultLI()
# Test.add_all([1,1,2,3],[9,8,9,8],[0.1,0.2,0.3,0.4],[0.01,0.02,0.005,0.002])
# Test.add_position(0)
# Test.add_insertion_time(0.005)
# Test.add_sensitivities([0.02,0.004,0.1])

# Test.save_to_json('test.json')
# New = json_to_obj('test.json', obj='LI')
# print(New.pos)


def train_classical(model, traindataloader,testdataloader, optimizer, no_epochs, scheduler, stopping_criterion=None, save_grad_norms=False):
    losses, test_accs,train_accs, times, grad_norms_layerwise = train(model, traindataloader,testdataloader, optimizer, no_epochs, scheduler, stopping_criterion=stopping_criterion, save_grad_norms=save_grad_norms)
    Res = ResultClassical()
    Res.add_all(losses, test_accs,train_accs, times, grad_norms_layerwise)
    return Res
