import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys

from smoothers import moving_average

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
sys.path.append('/export/home/lkreis/experiments_with_cnns/sensitivity-based-LI-for-cnns/code')
from training_oo import ResultClassical, ResultLI


def averaged_Result(list_of_Res, LI =False):
    Res_avg = ResultLI()

    losses = []
    accs = []

    for Res in list_of_Res:
        # add runs as rows
        losses.append(Res.losses)
        accs.append(Res.testaccs)

    a = np.array(losses)
    b = np.array(accs)

    losses_avg = np.nanmean(a, axis=0)
    accs_avg = np.nanmean(b, axis=0)

    Res_avg.add_losses(list(losses_avg))
    Res_avg.add_testaccs(list(accs_avg))
    if LI:
        Res_avg.add_insertion_epoch(list_of_Res[0].insertion_epoch)
    # TODO add times as well?
    return Res_avg

def plot_layerwise_grads_vgg_baseline(Res, moving_avg = None, BN=True, full_grad=False):
    g = Res.grad_norms
    l = len(g)
    gg = list(np.zeros_like(g[0]))
    if BN:
        labels = ['F1','b1', 'bns1','bnm1','F2','b2', 'bns2','bnm2','F3','b3', 'bns3','bnm3','W1','bf1', 'W2','bf2','Wlast']
        dims_layer = [3*64*3*3, 64, 64, 64, 64*128*3*3, 128, 128, 128, 128*256*3*3, 256, 256, 256, 256*4*4*500, 500, 500*500, 500, 500*10]
    else:
        labels = ['F1','b1', 'F2','b2','F3','b3','W1','bf1', 'W2','bf2','Wlast']
        dims_layer = [3*64*3*3, 64, 64*128*3*3, 128, 128*256*3*3, 256, 256*4*4*500, 500, 500*500, 500, 500*10]
    # todo calc gg
    for i in range(l):
        gg = [gg[j]+ g[i][j]*dims_layer[i] for j in range(len(gg))] 
    gg = [gg[j]/sum(dims_layer) for j in range(len(gg))]

    if moving_avg is not None:
        for i in range(l):
            g[i] = moving_average(g[i], moving_avg)
        if full_grad:
            gg = moving_average(gg, moving_avg)
    plt.figure(figsize=(20, 5))
    for i in range(l):
        plt.plot(g[i], label=labels[i])#f'0_{i}')
    if full_grad:
        plt.plot(gg,label='full grad')
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('layerwise gradient norms scaled by lr')
    plt.title('Baseline parameters ')
    plt.show()

def plot_layergrads_fullyext_model(ResLI, moving_avg=None, BN = True):
    styles = ['solid','solid','solid','solid', 'dotted','dotted','dotted','dotted','dashed','dashed','dashed','dashed','dashdot','dashdot','dashdot','dashdot']
    styles = styles + styles
    if BN:
        labels = ['F1','b1', 'bns1','bnm1','F11','b11', 'bns11','bnm11','F2','b2', 'bns2','bnm2','F22','b22', 'bns22','bnm22','F3','b3', 'bns3','bnm3','F33','b33', 'bns33','bnm33','W1','bf1', 'W2','bf2','Wlast']
        dims_layer = [3*64*3*3, 64, 64, 64,64*64*3*3, 64,64,64, 64*128*3*3, 128, 128, 128, 128*128*3*3, 128, 128, 128, 128*256*3*3, 256, 256, 256, 256*256*3*3, 256, 256,256, 256*4*4*500, 500, 500*500, 500, 500*10]
    else:
        labels = ['F1','b1', 'F11','b11', 'F2','b2', 'F22','b22', 'F3','b3', 'F33','b33', 'W1','bf1', 'W2','bf2','Wlast']
        dims_layer = [3*64*3*3, 64, 64*64*3*3, 64, 64*128*3*3, 128, 128*128*3*3, 128, 128*256*3*3, 256, 256*256*3*3, 256, 256*4*4*500, 500, 500*500, 500, 500*10]
    
    g = ResLI.ext_model_grads
    l = len(g)
    if moving_avg is not None:
        for i in range(l):
            g[i] = moving_average(g[i], moving_avg)
    plt.figure(figsize=(20, 5))
    for i in range(l):
        y = [1/dims_layer[i]*g[i][j] for j in range(len(g[i]))]
        plt.plot(y, label=labels[i],linestyle = styles[i])
    plt.legend()
    plt.yscale('log')
    plt.xlabel('iterations')
    plt.ylabel('layerwise squared gradient norms')
    plt.title('Sensitivity computation on fully extended net')
    plt.show()




def plot_layerwise_grads_vgg_LI(Res, moving_avg = None, BN=True, no_bias=False, save = None):
    if BN: lb=17
    else: lb= 11
    g1 = Res.grad_norms[0:lb]
    g2 = Res.grad_norms[lb:]
    colors = ['r','r','pink','pink', 'y', 'y','g','g','c', 'c','m' ]
    colors2 = ['r','r','b','b','pink','pink', 'y', 'y','g','g','c', 'c','m' ]
    if BN:
        labels1 = ['F1','b1', 'bns1','bnm1','F2','b2', 'bns2','bnm2','F3','b3', 'bns3','bnm3','W1','bf1', 'W2','bf2','Wlast']
        labels2 = ['F1','b1', 'bns1','bnm1','F2','b2', 'bns2','bnm2','F3','b3', 'bns3','bnm3','F4','b4', 'bns4','bnm4','W1','bf1', 'W2','bf2','Wlast']
    else:
        labels1 = ['F1','b1', 'F2','b2','F3','b3','W1','bf1', 'W2','bf2','Wlast']
        labels2 = ['F1','b1', 'F2','b2', 'F3','b3', 'F4','b4', 'W1','bf1', 'W2','bf2','Wlast']
    lb2 = len(g2)
    x_it1 = list(range(1,len(g1[0])+1))
    x_it2 = list(range(len(g1[0])+1,len(g1[0])+1 + len(g2[0])))
    if moving_avg is not None:
        for i in range(lb):
            g1[i] = moving_average(g1[i], moving_avg)
        for i in range(lb2):
            g2[i] = moving_average(g2[i], moving_avg)
    plt.figure(figsize=(20, 5))
    for i in range(lb-1):
        if no_bias and i%2==1:
            continue
        else:
            plt.plot(x_it1,g1[i],colors[i], label=labels1[i])#f'0_{i}')
    plt.vlines(x_it1[-1],ymin= 10e-5, ymax=10e-1, linestyles='dotted', colors='blue')
    for j in range(lb2-1):
        if no_bias and j%2==1:
            continue
        else:
            if colors2[j] == 'b':
                plt.plot(x_it2,g2[j],'b', label='Wnew')
            else:
                plt.plot(x_it2,g2[j],colors2[j])#, label=labels2[j])#f'0_{i}')
    plt.legend(fontsize=20, loc='lower right')
    plt.yscale('log')
    plt.xlabel('iterations', fontsize=20)
    plt.ylabel('kernel/weight gradient norms', fontsize=20)
    #plt.title(f'Layer insertion parameters with sens {Res.sensitivities}')
    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    plt.show()



def gen_avg_grad_norm_sq(Res):
    g_tot=0
    g_all = Res.grad_norms
    len_g_all = len(g_all)
    g_curr = []
    for g in g_all:
        g_curr.append(g)
    g_tot = np.nanmean(np.array(g_curr), axis=0)
    return g_tot

def plot_wrt_its(list_of_Res,LIs=[], labels = None, moving_avg = None, its_per_epoch = 391): # 391= roundup(50000/128)
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('xtick', labelsize=16)
    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
    fig.set_size_inches((20,10))
    colors_ = ['b', 'r', 'y','g', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink']
    if labels is None:
        labels = [f'{i}' for i in range(len(list_of_Res))]
    for i, Res in enumerate(list_of_Res):
        if moving_avg is not None:
            losses = moving_average(Res.losses, moving_avg)
        else:
            losses = Res.losses
        axes[0].plot(list(range(1,len(losses)+1)),losses, colors_[i], label = labels[i])
        if i in LIs:
            insertion_it = Res.insertion_epoch * its_per_epoch
            axes[0].vlines(insertion_it,min(losses),max(losses),linestyles='dotted',colors=colors_[i], label=labels[i])
    axes[0].legend(fontsize=15)
    axes[0].set_xlabel('iterations', fontsize=20)
    axes[0].set_ylabel(' loss', fontsize=20)
    axes[0].set_yscale('log')

    for i, Res in enumerate(list_of_Res):
        accs = Res.testaccs #trainaccs or testaccs
        axes[1].plot(list(range(1,len(accs)+1)),accs, colors_[i], label = labels[i])
        if i in LIs:
            axes[1].vlines(Res.insertion_epoch,60,100,linestyles='dotted',colors=colors_[i], label=labels[i])
    axes[1].legend(fontsize=15)
    axes[1].set_ylim(bottom=60)
    axes[1].set_xlabel('epochs', fontsize=20)
    axes[1].set_ylabel('test accuracy (%)', fontsize=20)

    plt.tight_layout()
    #plt.savefig(f'../../papers/plots/comp-fixed-architecture-{net_type}-loss-and-error.pdf', format="pdf", bbox_inches="tight")
    plt.show()

def plot_wrt_its_multipleli(list_of_Res,LIs=[], labels = None, moving_avg = None, its_per_epoch = 391, save = None, colors_=None): # 391= roundup(50000/128)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('xtick', labelsize=18)
    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
    fig.set_size_inches((20,10))
    if colors_ is None:
        colors_ = ['b', 'r', 'y','g', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink']
    if labels is None:
        labels = [f'{i}' for i in range(len(list_of_Res))]
    for i, Res in enumerate(list_of_Res):
        if moving_avg is not None:
            losses = moving_average(Res.losses, moving_avg)
        else:
            losses = Res.losses
        axes[0].plot(list(range(1,len(losses)+1)),losses, colors_[i], label = labels[i])
        if i in LIs:
            for j in Res.insertion_epochs:
                insertion_it = j * its_per_epoch
                axes[0].vlines(insertion_it,min(losses),max(losses),linestyles='dotted',colors='blue')#colors_[i])#, label=labels[i])
    axes[0].legend(fontsize=20)
    axes[0].set_xlabel('iterations', fontsize=20)
    axes[0].set_ylabel(' (minibatch) loss', fontsize=20)
    axes[0].set_yscale('log')

    for i, Res in enumerate(list_of_Res):
        accs = Res.testaccs #trainaccs or testaccs
        axes[1].plot(list(range(1,len(accs)+1)),accs, colors_[i], label = labels[i])
        if i in LIs:
            for j in Res.insertion_epochs:
                axes[1].vlines(j,60,100,linestyles='dotted',colors='blue')#, label=labels[i])
    axes[1].legend(fontsize=20)
    axes[1].set_ylim(bottom=60, top=90)
    axes[1].set_xlabel('epochs', fontsize=20)
    axes[1].set_ylabel('test accuracy (%)', fontsize=20)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    
    plt.show()

def plot_wrt_its_multipleli_only_loss(list_of_Res,LIs=[], labels = None, moving_avg = None, its_per_epoch = 391, save = None, colors_=None): # 391= roundup(50000/128)
    matplotlib.rc('ytick', labelsize=18)
    matplotlib.rc('xtick', labelsize=18)
    fig, axes = plt.subplots(1, gridspec_kw={'height_ratios': [5]})
    fig.set_size_inches((20,5))
    if colors_ is None:
        colors_ = ['b', 'r', 'y','g', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink']
    if labels is None:
        labels = [f'{i}' for i in range(len(list_of_Res))]
    for i, Res in enumerate(list_of_Res):
        if moving_avg is not None:
            losses = moving_average(Res.losses, moving_avg)
        else:
            losses = Res.losses
        axes.plot(list(range(1,len(losses)+1)),losses, colors_[i], label = labels[i])
        if i in LIs:
            for j in Res.insertion_epochs:
                insertion_it = j * its_per_epoch
                axes.vlines(insertion_it,min(losses),max(losses),linestyles='dotted',colors='blue')#colors_[i])#, label=labels[i])
    axes.legend(fontsize=20)
    axes.set_xlabel('iterations', fontsize=20)
    axes.set_ylabel('(minibatch) loss', fontsize=20)
    axes.set_yscale('log')

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    
    plt.show()


def plot_wrt_time(list_of_Res, LIs=[], labels= None, moving_avg = None, its_per_epoch=391, save = None):
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('xtick', labelsize=16)
    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
    fig.set_size_inches((20,10))
    if colors_ is None:
        colors_ = ['b', 'r', 'y','g', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink']
    if labels is None:
        labels = [f'{i}' for i in range(len(list_of_Res))]
    for i, Res in enumerate(list_of_Res):
        if moving_avg is not None:
            losses = moving_average(Res.losses, moving_avg)
        else:
            losses = Res.losses
        axes[0].plot(Res.times,losses, colors_[i], label = labels[i])
        if i in LIs:
            axes[0].vlines(Res.insertion_time,min(losses),max(losses),linestyles='dotted',colors=colors_[i], label=labels[i])
    axes[0].legend(fontsize=15)
    axes[0].set_xlabel('time(s)', fontsize=20)
    axes[0].set_ylabel(' loss', fontsize=20)
    axes[0].set_yscale('log')

    for i, Res in enumerate(list_of_Res):
        accs = Res.testaccs
        times_epoch = [Res.times[i*its_per_epoch -1] for i in range(1,int(len(Res.times)/its_per_epoch)+1)]
        axes[1].plot(times_epoch,accs, colors_[i], label = labels[i])
        if i in LIs:
            axes[1].vlines(Res.insertion_time,60,100,linestyles='dotted',colors=colors_[i], label=labels[i])
    axes[1].legend(fontsize=15)
    axes[1].set_ylim(bottom=60)
    axes[1].set_xlabel('time(s)', fontsize=20)
    axes[1].set_ylabel('test accuracy (%)', fontsize=20)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    plt.show()

def plot_wrt_time_multipleli(list_of_Res, LIs=[], labels= None, moving_avg = None, its_per_epoch=391, save = None, colors_=None):
    matplotlib.rc('ytick', labelsize=16)
    matplotlib.rc('xtick', labelsize=16)
    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios': [5,5]})
    fig.set_size_inches((20,10))
    if colors_ is None:
        colors_ = ['b', 'r', 'y','g', 'c', 'm', 'k', 'orange', 'purple', 'brown', 'pink']
    if labels is None:
        labels = [f'{i}' for i in range(len(list_of_Res))]
    for i, Res in enumerate(list_of_Res):
        if moving_avg is not None:
            losses = moving_average(Res.losses, moving_avg)
        else:
            losses = Res.losses
        axes[0].plot(Res.times,losses, colors_[i], label = labels[i])
        if i in LIs:
            for j in Res.insertion_times:
                axes[0].vlines(j,min(losses),max(losses),linestyles='dotted',colors='blue')#, label=labels[i])
    axes[0].legend(fontsize=15)
    axes[0].set_xlabel('time(s)', fontsize=20)
    axes[0].set_ylabel('(minibatch) loss', fontsize=20)
    axes[0].set_yscale('log')

    for i, Res in enumerate(list_of_Res):
        accs = Res.testaccs
        times_epoch = [Res.times[i*its_per_epoch -1] for i in range(1,int(len(Res.times)/its_per_epoch)+1)]
        axes[1].plot(times_epoch,accs, colors_[i], label = labels[i])
        if i in LIs:
            for j in Res.insertion_times:
                axes[1].vlines(j,60,100,linestyles='dotted',colors=colors_[i])#, label=labels[i])
    axes[1].legend(fontsize=15)
    axes[1].set_ylim(bottom=60, top=90)
    axes[1].set_xlabel('time(s)', fontsize=20)
    axes[1].set_ylabel('test accuracy (%)', fontsize=20)

    plt.tight_layout()
    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    plt.show()