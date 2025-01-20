import json
import sys
import matplotlib.pyplot as plt


##############################################################
##################### NOT useaböle any more ##################
##############################################################


sys.path.append('/home/leonie/codes/sensitivity-based-LI-for-cnns/code/plotting')

import smoothers


k = 8

with open(f"results_data/old_experiments/Exp{k}_1.json") as file:
    f = json.load(file)
    #print(f.keys())
    positions = [f[i]['positions'] for i in list(f.keys())] 
    #f.popitem()
    losses = [f[i]['losses'] for i in f.keys()]

print(f'{len(positions)} trainingruns have been performed!')

p0 = positions.count(0)
p1 = positions.count(1)
p2 = positions.count(2)

print(f'Position 0 is chosen {p0} times.')
print(f'Position 1 is chosen {p1} times.')
print(f'Position 2 is chosen {p2} times.')

for j in range(len(losses)):
    lo = losses[j]
    lo = smoothers.moving_average(lo,100)
    plt.plot(lo)
plt.show()