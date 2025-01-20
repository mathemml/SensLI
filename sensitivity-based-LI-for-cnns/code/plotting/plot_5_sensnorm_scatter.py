import averaging
from averaging import plot_wrt_its, plot_wrt_time
import sys
from matplotlib import pyplot as plt
import numpy as np

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
from training_oo import json_to_obj

k = 5
i = 1 # method no 
no_inits = 1

def plot_event_sensnorms(k,i, inits, save = None):

    no_norms = 5
    no_inits = len(inits)

    p0 = np.empty((no_norms,no_inits))
    p1 = np.empty((no_norms,no_inits))
    p2 = np.empty((no_norms,no_inits))



    for j, jj in zip(inits,range(no_inits)):
        
        if True:
            path = f"/home/leonie/codes/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_{i}_{j}.json"
            Res = json_to_obj(path, 'LI')
            print(f' for method {i}:\n')
            #print(f'chosen position: {Res.pos} ')
            print('based on norm op22')
            print('sens ordering: fro_squared, fro_squared_scaled, op22_1, op22_2, op22')
            #print(f'sens at pos0: {Res.sensitivities_all[0]}')
            #print(f'sens at pos1: {Res.sensitivities_all[1]}')
            #print(f'sens at pos2: {Res.sensitivities_all[2]}')
            for n in range(5):
                p0[n,jj] = Res.sensitivities_all[0][0][n]
                p1[n,jj] = Res.sensitivities_all[0][1][n]
                p2[n,jj] = Res.sensitivities_all[0][2][n]
        
    x = [2,4,6,8,10]
    # plot:
    fig, ax = plt.subplots()

    #pi have the shape (5,10)
    ax.eventplot(p0, orientation="vertical", lineoffsets=x, linewidth=1,linelength=0.5, colors='b', label='pos0')
    ax.eventplot(p1, orientation="vertical", lineoffsets=x, linewidth=1,linelength=0.5, colors='g', label='pos1')
    ax.eventplot(p2, orientation="vertical", lineoffsets=x, linewidth=1,linelength=0.5, colors='r', label='pos2')

    ax.set(xlim=(0,12), xticks=np.arange(1, 12), xticklabels=['','fro sq','','fro sq scaled','','op22_1','','op22_2','','op22',''], 
        xlabel='norm types', ylabel='sensitivity norm values')#, title=f'Comparing choices of sensitivity norms for exp{k} method {i}')
    ax.set_yscale('log')

    # Decrease font size of xticklabels and ylabel
    ax.set_xticklabels(['', 'fro sq', '', 'fro sq scaled', '', 'op22_1', '', 'op22_2', '', 'op22', ''], fontsize=10)
    ax.tick_params(axis='y', labelsize=10)

    #ax.legend(['fro squared','fro squared scaled','op22_1','op22_2','op22'], loc='upper right')
    #plt.legend()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower right')

    if save is not None:
        plt.savefig(save, format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    for i in [1]:
        plot_event_sensnorms(k,i, [0]) 
