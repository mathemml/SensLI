import torch
import time
import utils
############################################
import numpy as np
import os
############################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, traindataloader,testdataloader, optimizer, no_epochs, scheduler, stopping_criterion=None, save_grad_norms=True, kfac_precond =None):
    ################################################################################################
    save_heatmaps = False
    if save_heatmaps:
        if save_heatmaps:
            heatmappathgrads=f'heatmaps/'
            #os.mkdir(heatmappathgrads)
            
            heatmappathvals=f'heatmaps/'
            #os.mkdir(heatmappathvals)
    ################################################################################################

    # TODO stopping criterion
    if stopping_criterion is not None:
        raise Exception('Sorry, stopping criteria are not implemented yet!')
    
    model.to(device)
    times = []
    losses = []
    test_accs = []
    train_accs = []
    grad_norms = []
    grad_norms_layerwise = []
    for p in model.parameters():
        grad_norms_layerwise.append([])
    if kfac_precond is not None:
        precond_epochs = 50

    for e in range(no_epochs):
        #toc = time.time()
        print('\nEpoch: %d' % e)
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(traindataloader):
            toc = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            ################################################################################################
            if save_heatmaps and batch_idx<3:
                # save heatmaps of gradients and values
                #save_weightgrads_heatmap(model, batch, e, path=heatmappathgrads)
                #save_weightvalues_heatmap(model, batch, e, path=heatmappathvals)
                #save weights and grads for heatmapplotting later
                with torch.no_grad():
                    for i, p in enumerate(model.parameters()):
                        if i in [2]:
                            filename = f'{heatmappathvals}vals_epoch{e}_batch{batch_idx}_param{i}.txt'
                            np.savetxt(filename, p.data.detach().view(30,30).numpy())
                            filename = f'{heatmappathgrads}grads_epoch{e}_batch{batch_idx}_param{i}.txt'
                            np.savetxt(filename, p.grad.data.detach().view(30,30).numpy())
            ################################################################################################
            with torch.no_grad(): # NEW
                if save_grad_norms: 
                    norm = 0
                    layer = 0
                    lr = optimizer.param_groups[0]['lr']
                    for p in model.parameters():
                        grad_norms_layerwise[layer].append(
                            lr*torch.square(p.grad).cpu().sum().numpy())
                        layer += 1
                    for p in model.parameters():
                        norm += torch.square(p.grad).sum()
                    grad_norms.append(norm)
                    losses.append(loss.item()) 
            # we replace: optimizer.step() with..
            with torch.no_grad():
                for p in model.parameters():
                    if kfac_precond is not None and e<precond_epochs and p is kfac_precond[0] :
                        p-= optimizer.param_groups[0]['lr'] * utils.precond(kfac_precond[1], p.grad)
                    else:
                        p-= optimizer.param_groups[0]['lr'] * p.grad
            model.zero_grad()

            #train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            tic=time.time()
            if len(times)>0:
                times.append(tic-toc+ times[-1])
            else:
                times.append(tic-toc)
        #train_loss=train_loss/(batch_idx+1)
        #losses.append(train_loss)
        acc = 100.*correct/total
        print(f'training accuracy: {acc} and loss {loss.item()}') 
        test_acc = test(model, testdataloader)
        train_acc = test(model, traindataloader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f'test accuracy: {test_acc}')
        #tic=time.time()
        # old time keeping
        # check if scheduler is reduce on plateau
        if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            scheduler.step(test_acc) # train_acc and test_acc and 1/loss are possible
        else:
            scheduler.step()  


    return losses, test_accs,train_accs, times, grad_norms_layerwise


def test(model,testdataloader):

    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testdataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
    return acc