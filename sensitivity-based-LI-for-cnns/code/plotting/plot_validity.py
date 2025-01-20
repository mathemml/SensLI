import averaging
import sys

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
from training_oo import json_to_obj

k = 9
i = 1 # method no # start from 1
j = 0 # training run # start from 0
no_runs = 1


###################### averaged plotting #############################################

method_types = ['LI','LI','Classical','Classical','Classical','Classical','LI']#,'LI','LI']
LIs = [0,1,6]#,7,8]
labels = ['SensLI', 'LIother','Baseline','CNN0','CNN1','CNN2','LImiddle']#,'LIreltr','LIabstr']


list_of_avg_Res = []
LIs_local = []
labels_local = []


for i in [1,2]:#,3,4,5,6,7]:
    if i-1 in LIs: LI = True 
    else: LI=False
    list_of_runs = []
    
    for j in range(no_runs):
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
save = f'code/plotting/plot_figs/validity-li-cnn-loss-and-error-its.pdf'
colors = ['b','orange']

averaging.plot_wrt_its_multipleli_only_loss(list_of_avg_Res, LIs_local, labels_local, moving_avg=50, save = save, colors_=colors)

# # every single run wrt its

# for j in range(10):
#     l = []
#     for i in range(1,7):#10):
#         Res = json_to_obj(f"/home/leonie/codes/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_{i}_{j}.json", method_types[i-1])
#         l.append(Res)
#     #averaging.plot_wrt_its(l, LIs, labels, moving_avg=10)

