import averaging
import sys

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
from training_oo import json_to_obj

k = 12 
i = 1 # method no # start from 1
j = 0 # training run # start from 0


###################### averaged plotting #############################################

method_types = ['LI','LI','LI','LI','LI','LI','LI','LI','Classical']
LIs = [0,1,2,3,4,5,6,7]
labels = ['LI10', 'LI20', 'LI30', 'LI40', 'LI50', 'LI60', 'LI70', 'LI80', 'CNN1']


list_of_avg_Res = []
LIs_local = []
labels_local = []


for i in [1,2,3,4,5,6,7,8,9]:
    print(i)
    if i-1 in LIs: LI = True 
    else: LI=False
    list_of_runs = []
    
    for j in range(1):
        Res = json_to_obj(f"/home/leonie/codes/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_{i}_{j}.json", method_types[i-1])
        if LI:
            print(Res.positions)
            print(Res.sensitivities)
            print(Res.ext_full_grad)
        list_of_runs.append(Res)
    if LI:
        LIs_local.append(i-1)
    labels_local.append(labels[i-1])

    Res_avg = Res#averaging.averaged_Result(list_of_runs, LI)
    
    list_of_avg_Res.append(Res_avg)

# plot average wrt its

save = f'code/plotting/plot_figs/when-to-insert-cnn-loss-and-error-its.pdf'

averaging.plot_wrt_its_multipleli(list_of_avg_Res, LIs_local, labels_local, moving_avg=170, save = save)

