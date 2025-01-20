import averaging
import sys

# setting path
sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code')
from training_oo import json_to_obj
from plot_30_sensnorm_scatter import plot_event_sensnorms

k = 5
# i = 1 # method no # start from 1
# j = 0 # training run # start from 0
no_runs=10
inits = list(range(no_runs))

###################### averaged plotting #############################################

method_types = ['LI','LI','Classical','Classical','Classical','Classical','LI']#,'LI','LI']
LIs = [0,1,6]#,7,8]
labels = ['CNN LI', 'LImin','CNN1','CNN2','CNN1','CNN2','LImiddle']#,'LIreltr','LIabstr']


list_of_avg_Res = []
LIs_local = []
labels_local = []


for i in [1,3,4]:#,5,6,7]:
    if i-1 in LIs: LI = True 
    else: LI=False
    list_of_runs = []
    
    for j in range(1, no_runs):
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
save = f'code/plotting/plot_figs/comp-fixed-arch-cnn-loss-and-error-its.pdf'

averaging.plot_wrt_its_multipleli(list_of_avg_Res, LIs_local, labels_local, moving_avg=170, save = save)


# plot grads

i=1
j=0
path = f"/home/leonie/codes/sensitivity-based-LI-for-cnns/results_data/Exp{k}/Exp{k}_{i}_{j}.json"
Res = json_to_obj(path, method_types[i-1]) 
print(Res.positions)
save_loss = "code/plotting/plot_figs/comp-fixed-arch-cnn-loss-and-error-singleloss.pdf"
averaging.plot_wrt_its_multipleli_only_loss([Res],[0],['SensLI'],  moving_avg=100,save=save_loss)
save_grads = 'code/plotting/plot_figs/comp-fixed-arch-cnn-loss-and-error-grads.pdf'
averaging.plot_layerwise_grads_vgg_LI(Res, moving_avg=100, BN=False, no_bias=True, save = save_grads)

# plot sensnorm scatter plot
save_sensnorm ='code/plotting/plot_figs/comp-fixed-arch-cnn-loss-and-error-sensnorms.pdf'
plot_event_sensnorms(k,1,inits,save=save_sensnorm)
