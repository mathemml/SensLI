import averaging
import sys

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
from training_oo import json_to_obj


k=1
j=0 # only one run

#################################################

method_types = ['LI','LI','Classical','Classical', 'Classical','Classical']
LIs = [0]
labels = ['SensLI','CNN']#['LI','LI min','Baseline','big CNN 1', 'big CNN 2'] #['LI','CNN2']

list_of_methods = []

for i in [1,4]:
#for i in [1,5]:
    path = f"/home/leonie/codes/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_{i}_{j}.json"
    Res = json_to_obj(path, method_types[i-1])
    if i-1 in LIs:
        print(Res.positions)
        #print(Res.sensitivities)
        #print(Res.ext_full_grad)
    list_of_methods.append(Res)


save = f'code/plotting/plot_figs/performance-cnn-loss-and-error-its.pdf'
save_time = f'code/plotting/plot_figs/performance-cnn-loss-and-error-time.pdf'
colrs = ['b','y']

averaging.plot_wrt_its_multipleli(list_of_methods, LIs, labels, moving_avg=170, save=save, colors_=colrs)


averaging.plot_wrt_time_multipleli(list_of_methods, LIs, labels, moving_avg=170, save=save_time, colors_=colrs)



#for 1: sensLI inserts positions 0,-,0-> 64x64, -, 64x64 inserted
