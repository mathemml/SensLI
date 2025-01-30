import torch
import time

from model_selection_oo import VGGExtendableModel, SmallVGGExtendableModel
from train_and_test_ import train
from training_oo import ResultLI


                        
                        
    
def training_with_one_LI(epochs, traindataloader, testdataloader,BN=False, optimizer_type='SGD', lr_init=0.1,
                         mode='abs max', stopping_criterion=None,lrschedule_type='StepLR', lrscheduler_args=None,
                         decrease_lr_after_li=1.,save_grad_norms=True, init=None, use_kfac = False, sens_norm = 'all',
                         architecture='VGG', m=[64,128,256], max_LIs=1):
    
    """
    SensLI methodfor CNNs.

    Parameters:
    epochs (list of int): Number of epochs to train the model.
    traindataloader (DataLoader): DataLoader for the training data.
    testdataloader (DataLoader): DataLoader for the test data.
    BN (bool): Whether to use Batch Normalization. Default is False. Not everything is supported for BatchNorm. Use only if you know the code.
    optimizer_type (str): Type of optimizer to use ('SGD' or 'Adam'). Default is 'SGD'.
    lr_init (float): Initial learning rate. Default is 0.1.
    mode (str): Mode for layer insertion. Default is 'abs max'. 'thresold_rel' is used for the when-to-insert-heuristic.
    stopping_criterion (callable): Function to determine when to stop training. Default is None.  Not implemented yet.
    lrschedule_type (str): Type of learning rate scheduler ('StepLR', 'MultiStepLR', 'ReduceLROnPlateau'). Default is 'StepLR'.
    lrscheduler_args (dict): Arguments for the learning rate scheduler. Default is None.
    decrease_lr_after_li (float): Factor to decrease learning rate after layer insertion. Default is 1.0.
    save_grad_norms (bool): Whether to save gradient norms. Default is True.
    init (Tensor): Initial parameters for the model as vector. Default is None.
    use_kfac (bool): Whether to use KFAC optimizer. Default is False. not fully supported in the code. Use only if you know the code.
    sens_norm (str): Sensitivity normalization method. Default is 'all'.
    architecture (str): Model architecture ('VGG' or 'VGGsmall'). Default is 'VGG'.
    m (list): List of integers specifying the number of filters in each layer. Default is [64, 128, 256].
    max_LIs (int): Maximum number of layer insertions. Default is 1.

    Returns:
    Res object of class ResultLI: Result object containing all relevant information.
    """
    
    if BN==True and use_kfac==True:
        raise NotImplementedError('combination of Batchnorm and KFAC not implemented')
    
    pos = None

    if architecture == 'VGG':
        VGG = VGGExtendableModel(m,BN, use_kfac=use_kfac)
    if architecture == 'VGGsmall':
        VGG = SmallVGGExtendableModel(m=m, BN=BN, use_kfac=use_kfac)
    Res = ResultLI()

    print(f'curr model: {VGG.model}')

    print(f'Starting training on baseline model...')
    if init is not None:
        torch.nn.utils.vector_to_parameters(init, VGG.model.parameters())

    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(VGG.model.parameters(), lr_init, momentum=0.9, weight_decay=5e-4)
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(VGG.model.parameters(), lr_init)

    # build lr scheduler
    if lrschedule_type == 'StepLR':
        if isinstance(lrscheduler_args['step_size'],list):
            step_size = lrscheduler_args['step_size'][0]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma)
            
    if lrschedule_type == 'MultiStepLR':
        if isinstance(lrscheduler_args['step_size'][0],list):
            step_size = lrscheduler_args['step_size'][0]
        else:
            step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=step_size, gamma=gamma)
        
    if lrschedule_type == 'ReduceLROnPlateau':
        step_size = lrscheduler_args['step_size']
        gamma = lrscheduler_args['gamma']
        lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=gamma, patience=step_size, verbose=False)
        
    ############## TRAIN BASELINE ######################################

    
        
    losses1, test_accs1, train_accs1, times1, grad_norms1 = train(VGG.model,traindataloader,
                                        testdataloader,optimizer,epochs[0],
                                        lrscheduler, stopping_criterion=stopping_criterion, 
                                        save_grad_norms=save_grad_norms)
    
    Res.add_all(losses1, test_accs1, train_accs1, times1, grad_norms1)

    if len(times1) == 0:
        time_for_baseline = 0
    else:
        time_for_baseline= times1[-1]

    no_LIs = 0

    while no_LIs < max_LIs: # if a layer is not inserted, try again

        ############# BUILD TMP NET ########################################
        print(f'Starting layer selection... for try {no_LIs}')
        toc = time.time()
        VGG.fully_extend()

        ######## COMPUTE SENSITIVITIES #####################################
        print('calculate sensitivities...')
        if use_kfac:  print(' and calculate kfac...')
        VGG.calculate_shadow_prices_mb(traindataloader)
        
        ####### SELECT NEW MODEL ###########################################
        print('build new model...')
        VGG.select_new_model(rule = mode, sens_norm = sens_norm)
        pos = VGG.pos 
        if VGG.use_kfac:
            if pos is not None:
                tuple_kfac = VGG.kfacs[pos]
                p_new = VGG.new_weight_param
                param_and_tuple_kfac = [p_new, tuple_kfac]
        else:
            param_and_tuple_kfac = None


        ##### DECREASE LR ##################################################
        # decrease only if pos not none
        lr_end = optimizer.param_groups[0]['lr']
        if pos is not None:
            lr_init_ext = decrease_lr_after_li * lr_end
        else:
            lr_init_ext = lr_end

        tic = time.time()

        time_model_sel_diff = tic- toc
        ######## TRAIN ON EXTENDED MODEL ###################################
        print(f'Start training again.....')

        if optimizer_type == 'SGD':
            optimizer = torch.optim.SGD(VGG.model.parameters(), lr_init_ext, momentum=0.9, weight_decay=5e-4)
        if optimizer_type == 'Adam':
            optimizer = torch.optim.Adam(VGG.model.parameters(), lr_init_ext)

        # build lr scheduler
        if lrschedule_type == 'StepLR':
            if isinstance(lrscheduler_args['step_size'],list):
                step_size = lrscheduler_args['step_size'][1]
            else:
                step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=step_size, gamma=gamma)
                
        if lrschedule_type == 'MultiStepLR':
            if isinstance(lrscheduler_args['step_size'][0],list):
                step_size = lrscheduler_args['step_size'][1]
            else:
                step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer, milestones=step_size, gamma=gamma)
        
        if lrschedule_type == 'ReduceLROnPlateau':
            step_size = lrscheduler_args['step_size']
            gamma = lrscheduler_args['gamma']
            lrscheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', factor=gamma, patience=step_size, verbose=False)

        
        losses2, test_accs2, train_accs2, times2, grad_norms2 = train(VGG.model,traindataloader,
                                            testdataloader,optimizer,epochs[no_LIs+1],
                                            lrscheduler, stopping_criterion=stopping_criterion,
                                            save_grad_norms=save_grad_norms,
                                            kfac_precond = param_and_tuple_kfac)
        
        if len(times2)>0:
            times2 = [times2[i]+ time_model_sel_diff + time_for_baseline for i in range(len(times2))]

        Res.add_all(losses2, test_accs2,train_accs2, times2,grad_norms2)

        no_LIs += 1
        if len(times2) == 0:
             time_for_baseline = 0
        else:
            time_for_baseline = times2[-1]

        if pos is not None:
            insertion_epoch = sum(epochs[0:no_LIs])
        # add position
            Res.add_position(pos)
        # add insertion time
            Res.add_insertion_time(times2[0])
        # add insertion epoch
            Res.add_insertion_epoch(insertion_epoch)
        # add sensitivities
            Res.add_sensitivities(VGG.sensitivities)
        # add sens for all norms and use deafault::
            if sens_norm == 'all' and mode == 'abs max':
                Res.add_sensitivities_all(VGG.sensitivities_all)
        # add full grad of ext model
            if use_kfac:
                Res.add_ext_full_grad(VGG.ext_full_grad_sq_scaled)
            else:
                Res.add_ext_full_grad(VGG.ext_full_grad_sq_scaled.item())
        # add layerwise grads of ext model during sensitivity computation
            Res.add_grads_of_ext_model(VGG.ext_model_layer_grads)
        
    if pos is None:
        Res.add_position(None)
        Res.add_insertion_epoch(None)
        Res.add_insertion_time(None)
        Res.add_sensitivities(VGG.sensitivities)
        Res.add_ext_full_grad(None)
        Res.add_grads_of_ext_model(VGG.ext_model_layer_grads)

    print(f'LI Training finished!')
    return Res


